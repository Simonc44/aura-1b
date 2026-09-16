"""Niveau 0 : reponse instantanee (0.1s) — le coeur de l'architecture CPU-native.

Philosophie : sur un PC sans GPU, le composant le plus lent est le LLM
(~14 tok/s). La reponse la plus rapide est celle qu'on n'a pas besoin de
generer. Deux detecteurs, du plus rapide au plus cher :

1. **Calcul direct** (< 1 ms) — regex + eval arithmetique securise (AST) :
   « carre de 12 », « 2 puissance 10 », « 45*12 », « 15% de 200 »...
   Resultat EXACT, zero hallucination, zero token genere.
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

# -- eval arithmetique securise (AST, jamais eval()) ------------------------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_PUISSANCE_MAX = 1e15  # garde-fou : 2**10000 ferait planter le CPU


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

    # 2. cache semantique (~1 ms)
    _charger_cache()
    if _entrees:
        i = _similar(question)
        if i is not None:
            LOG.info("[niveau0] cache semantique en %.1f ms",
                     (time.time() - t0) * 1000)
            return _entrees[i]["r"]
    return None


def enregistrer(question: str, reponse: str):
    """Memorise une reponse (apres generation LLM) pour les futures questions."""
    if not question or not reponse or reponse.startswith("[Aura]"):
        return
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


def stats() -> dict:
    _charger_cache()
    return {"entrees_cache": len(_entrees), "seuil": _SEUIL_SIMILARITE,
            "fichier": str(_FICHIER)}
