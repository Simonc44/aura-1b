"""Niveau 0 : reponse instantanee (0.1s) — le coeur de l'architecture CPU-native.

Philosophie : sur un PC sans GPU, le composant le plus lent est le LLM
(~14 tok/s). La reponse la plus rapide est celle qu'on n'a pas besoin de
generer. Deux detecteurs, du plus rapide au plus cher :

1. **Calcul direct** (< 1 ms) — regex + eval arithmetique securise (AST) :
   « carre de 12 », « 2 puissance 10 », « 45*12 », « 15% de 200 » et les
   expressions dictees en mots (« 15 divise par 3 plus 4 puissance 2 »).
   Resultat EXACT, zero hallucination, zero token genere.
1bis. **PAL** (< 1 ms) — dates (« dans 45 jours », « combien de jours
   jusqu'au 25 decembre »), unites (« 5 miles en km », « 100 f en c »),
   pourcentages composes (« 15% de 200 plus 30% de 100 ») : voir pal.py.
2. **Cache semantique** (~1 ms) — TF-IDF char n-grams + cosinus sur les
   reponses deja donnees : une question similaire (>= 0.85) renvoie la
   reponse enregistree. Le systeme « apprend » ses reponses.

Si rien ne matche → None → le routeur et le Llama prennent la main.
"""
import ast
import json
import logging
import operator
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

LOG = logging.getLogger("aura.filtre0")

_FICHIER = Path(__file__).parent / ".cache_reponses.jsonl"
_SEUIL_SIMILARITE = 0.85

# reponses contextuelles (liées a l'utilisateur ou a la conversation) :
# a ne JAMAIS mettre en cache, la reponse serait fausse hors contexte.
_MOTS_CONTEXTUELS = ("mon ", "ma ", "mes ", "je ", "j'", "moi", "tu ", "ton ",
                     "ta ", "tes ", "nous ", "votre ", "vos ", "prenom",
                     "mon nom", "mon age")

# -- eval arithmetique securise (AST, jamais eval()) ------------------------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_PUISSANCE_MAX = 1e15  # garde-fou : 2**10000 ferait planter le CPU

# -- expressions dictees en mots ---------------------------------------------
# « 15 divise par 3 plus 4 puissance 2 » -> « 15 / 3 + 4 ** 2 »
_MOTS_OPERATEURS = (
    (re.compile(r"\bdivis\w*\s+par\b"), "/"),
    (re.compile(r"\bmultipli\w*\s+par\b"), "*"),
    (re.compile(r"\bfois\b"), "*"),
    (re.compile(r"\bau\s+carre\b"), "**2"),
    (re.compile(r"\bau\s+cube\b"), "**3"),
    (re.compile(r"\bpuissance\b"), "**"),
    (re.compile(r"\bplus\b"), "+"),
    (re.compile(r"\bmoins\b"), "-"),
)
# operateurs « forts » : leur presence seule autorise la conversion
# (plus/moins seuls sont ambigus : « il fait moins 5 degres »)
_OP_FORTE = re.compile(
    r"\b(divis\w*\s+par|multipli\w*\s+par|fois|puissance|au\s+carre|au\s+cube)\b")
_DEMANDE_MATH = re.compile(
    r"\b(combien fait|combien vaut|calcule|resultat de|resous)\b")
_RESIDUS = re.compile(r"[\d.+\-*/()\s]+")
_NOMBRE = re.compile(r"\d+(?:\.\d+)?")


def _expression_en_mots(q: str) -> str | None:
    """Convertit une expression dictee en arithmetique, ou None.

    « 15 divise par 3 plus 4 puissance 2 » -> « 15 / 3 + 4 ** 2 ».
    Les mots non convertis sont jetes ; il faut >= 2 nombres et >= 1
    operateur binaire pour tenter l'evaluation.
    """
    expr = q
    for motif, op in _MOTS_OPERATEURS:
        expr = motif.sub(f" {op} ", expr)
    candidat = " ".join("".join(_RESIDUS.findall(expr)).split())
    if len(_NOMBRE.findall(candidat)) < 2:
        return None
    if not re.search(r"[+\-*/]", candidat.replace("**", "")):
        return None
    return candidat


def _evaluer(noeud):
    if isinstance(noeud, ast.Expression):
        return _evaluer(noeud.body)
    if isinstance(noeud, ast.Constant) and isinstance(noeud.value, (int, float)):
        return noeud.value
    if isinstance(noeud, ast.BinOp) and type(noeud.op) in _OPS:
        gauche, droite = _evaluer(noeud.left), _evaluer(noeud.right)
        if isinstance(noeud.op, ast.Pow):
            if abs(droite) > 64 or abs(gauche) ** min(abs(droite), 64) > _PUISSANCE_MAX:
                raise ValueError("puissance hors limites")
        return _OPS[type(noeud.op)](gauche, droite)
    if isinstance(noeud, ast.UnaryOp) and type(noeud.op) in _OPS:
        return _OPS[type(noeud.op)](_evaluer(noeud.operand))
    raise ValueError("expression non autorisee")


def _calcul_direct(question: str) -> str | None:
    """Resout les maths directes par regex + AST. Aucun LLM, aucune erreur."""
    q = unicodedata.normalize("NFKD", question.lower())
    q = "".join(c for c in q if not unicodedata.combining(c))
    q = q.replace(",", ".").replace("x", "*").replace("×", "*") \
         .replace("÷", "/").replace("^", "**")

    # expression dictee en mots, AVANT les motifs simples : sur une
    # expression composee, « 4 puissance 2 » tout seul donnerait 16
    # au lieu de 21 (le bug du 15/3 + 4^2)
    if _OP_FORTE.search(q) or _DEMANDE_MATH.search(q):
        expr = _expression_en_mots(q)
        if expr:
            try:
                return _fmt(_evaluer(ast.parse(expr, mode="eval")))
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                pass

    # « carre de 12 » / « cubé de 3 »
    m = re.search(r"carre de \(?(-?\d+(?:\.\d+)?)\)?", q)
    if m:
        n = float(m.group(1))
        return _fmt(n * n)

    # « racine carree de 144 »
    m = re.search(r"racine carree de \(?(\d+(?:\.\d+)?)\)?", q)
    if m:
        return _fmt(float(m.group(1)) ** 0.5)

    # « 2 puissance 10 »
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*puissance\s*(-?\d+)", q)
    if m and abs(int(m.group(2))) <= 64:
        return _fmt(float(m.group(1)) ** int(m.group(2)))

    # « 15% de 200 » / « 15 pourcent de 200 »
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|pourcent)\s*de\s*(\d+(?:\.\d+)?)", q)
    if m:
        return _fmt(float(m.group(1)) / 100 * float(m.group(2)))

    # expression arithmetique brute : « calcule 45*12+3 »
    m = re.search(r"(-?\d+(?:\.\d+)?(?:\s*[\+\-\*/%]\s*\(?-?\d+(?:\.\d+)?\)?)+)", q)
    if m:
        try:
            arbre = ast.parse(m.group(1), mode="eval")
            return _fmt(_evaluer(arbre))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            return None
    return None


def _fmt(v: float) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v:,}".replace(",", " ") if isinstance(v, int) else f"{v:.6g}"


# -- cache semantique --------------------------------------------------------

_vectoriseur = None
_matrice = None
_entrees: list[dict] = []


def _charger_cache():
    global _vectoriseur, _matrice, _entrees
    if _vectoriseur is not None:
        return
    if _FICHIER.exists():
        try:
            _entrees = [json.loads(l) for l in
                        _FICHIER.read_text(encoding="utf-8").splitlines() if l.strip()]
        except (OSError, json.JSONDecodeError):
            _entrees = []
    if _entrees:
        from sklearn.feature_extraction.text import TfidfVectorizer
        _vectoriseur = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                       max_features=2000)
        _matrice = _vectoriseur.fit_transform([e["q"] for e in _entrees])


def _similar(question: str) -> int | None:
    """Index de l'entree la plus similaire si >= seuil, sinon None."""
    from sklearn.metrics.pairwise import cosine_similarity
    if _vectoriseur is None or _matrice is None:
        return None
    vec = _vectoriseur.transform([question.lower().strip()])
    sims = cosine_similarity(vec, _matrice)[0]
    i = int(np.argmax(sims))
    return i if sims[i] >= _SEUIL_SIMILARITE else None


def repondre(question: str) -> str | None:
    """Tente une reponse instantanee. None = laisser le LLM prendre la main."""
    t0 = time.time()

    # 1. calcul direct (< 1 ms)
    r = _calcul_direct(question)
    if r is not None:
        LOG.info("[niveau0] calcul direct en %.1f ms", (time.time() - t0) * 1000)
        return r

    # 1bis. PAL (< 1 ms) : dates, unites, pourcentages composes —
    # import paresseux (pal importe filtre_instantane pour l'AST garde)
    from . import pal
    r = pal.repondre(question)
    if r is not None:
        LOG.info("[niveau0] PAL en %.1f ms", (time.time() - t0) * 1000)
        return r

    # 2. cache semantique (~1 ms) — JAMAIS pour une question contextuelle :
    # sa reponse depend de la conversation en cours, pas du texte seul.
    if any(m in question.lower() for m in _MOTS_CONTEXTUELS):
        return None
    _charger_cache()
    if _entrees:
        i = _similar(question)
        if i is not None:
            LOG.info("[niveau0] cache semantique en %.1f ms",
                     (time.time() - t0) * 1000)
            return _entrees[i]["r"]
    return None


def enregistrer(question: str, reponse: str):
    """Memorise une reponse (apres generation LLM) pour les futures questions.

    Les questions contextuelles (mon prenom, mon age...) sont exclues : leur
    reponse depend de la conversation, pas du texte de la question.
    """
    if not question or not reponse or reponse.startswith("[Aura]"):
        return
    q = question.lower().strip()
    if any(m in q for m in _MOTS_CONTEXTUELS):
        return  # reponse contextuelle : jamais en cache
    _charger_cache()
    if _entrees and _similar(question) is not None:
        return  # deja connu
    entree = {"q": question.lower().strip(), "r": reponse,
              "t": time.strftime("%Y-%m-%d")}
    _entrees.append(entree)
    try:
        with _FICHIER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entree, ensure_ascii=False) + "\n")
        from sklearn.feature_extraction.text import TfidfVectorizer
        global _vectoriseur, _matrice
        if _vectoriseur is None:
            _vectoriseur = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                           max_features=2000)
            _matrice = _vectoriseur.fit_transform([e["q"] for e in _entrees])
        else:
            nouvelle = _vectoriseur.transform([entree["q"]])
            from scipy.sparse import vstack
            _matrice = vstack([_matrice, nouvelle])
    except OSError as e:
        LOG.warning("[niveau0] cache non ecrit : %s", e)


def mettre_a_jour(question: str, reponse: str) -> bool:
    """RECONSOLIDATION : remplace la reponse (similaire) existante par la nouvelle.

    Comme le cerveau : un souvenir contredit par une preuve plus forte est
    reecrit, pas double. Sans cela, une reponse fausse mise en cache reste
    collée pour toujours (enregistrer refuse les doublons).
    Renvoie True si une entree a ete remplacee.
    """
    global _vectoriseur, _matrice
    if not question or not reponse or reponse.startswith("[Aura]"):
        return False
    _charger_cache()
    if _vectoriseur is None or _matrice is None:
        # cache vide (aucune memoire a reconsolider) : rien a remplacer.
        # Cas legitime : machine neuve, checkout CI sans .cache_reponses.jsonl.
        return False
    i = _similar(question.lower().strip())
    if i is None:
        return False
    _entrees[i]["r"] = reponse
    _vectoriseur, _matrice = None, None      # recalcule au prochain acces
    try:
        with _FICHIER.open("w", encoding="utf-8") as f:
            for e in _entrees:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        LOG.info("[niveau0] reconsolidation : reponse remplacee pour « %s »",
                 question[:50])
        return True
    except OSError as e:
        LOG.warning("[niveau0] reconsolidation non ecrite : %s", e)
        return False


def purger_contextuelles():
    """Retire du cache les entrees contextuelles (mises en cache avant le filtre)."""
    global _vectoriseur, _matrice, _entrees
    _charger_cache()
    avant = len(_entrees)
    garder = [e for e in _entrees
              if not any(m in e["q"] for m in _MOTS_CONTEXTUELS)]
    if len(garder) != avant:
        _entrees = garder
        _vectoriseur, _matrice = None, None   # recalcule au prochain acces
        try:
            with _FICHIER.open("w", encoding="utf-8") as f:
                for e in _entrees:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        except OSError as e:
            LOG.warning("[niveau0] purge non ecrite : %s", e)
        LOG.info("[niveau0] purge : %d entrees contextuelles retirees",
                 avant - len(garder))
    return avant - len(_entrees)


def stats() -> dict:
    _charger_cache()
    return {"entrees_cache": len(_entrees), "seuil": _SEUIL_SIMILARITE,
            "fichier": str(_FICHIER)}
