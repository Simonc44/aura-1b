"""Graphe de faits leger (inspire de MiniRAG, ACL 2026) : retrouver au lieu de deviner.

MiniRAG demontre qu'un petit modele reussit mieux quand il SUIV un chemin
dans un graphe de connaissances au lieu de generer de sa memoire interne.
Version minimale, sans dependance :

- stockage : triplets (sujet, relation, objet) dans un JSONL, un fichier
- lecture : index mots-cles en memoire, score = couverture des mots de la
  question (multi-mots = multi-hop naturel : une question sur « capitale de
  l australie » active le triplet via le sujet ET la relation)
- ecriture : extraction conservatrice de triplets depuis les reponses
  verifiees web (pattern « X est la capitale de Y », « X est le president
  de Y »...) — rien n'est extrait d'une reponse non verifiee
"""
import json
import logging
import os
import re
import threading
import time
import unicodedata

LOG = logging.getLogger("aura.graphe")

_FICHIER = os.path.join(os.path.dirname(__file__), ".graphe_faits.jsonl")
_index: dict[str, set[int]] = {}
_triplets: list[dict] = []
_chargé = False
_verrou = threading.Lock()

_STOPWORDS = frozenset(
    "le la les un une des de du au aux et ou ou est sont que qui quoi dont "
    "ou ni mais car donc pour par sur dans avec sans sous entre vers chez "
    "je tu il elle on nous vous ils elles me te se lui leur y en mon ton son "
    "ma ta sa mes tes ses notre votre nos vos leurs ceci cela ce cet cette "
    "ces quel quelle quels quelles combien comment pourquoi quand etre avoir "
    "a ai as avons avez ont suis es est sommes etes sont etais etait sera "
    "sera plus moins tres peu tout tous toute toutes meme aussi alors donc "
    "oui non voila voici la bas ici".split()
)

_MOTS = re.compile(r"[a-zàâäéèêëïîôöùûüç]{3,}")

# patterns d'extraction conservatrice (reponses deja verifiees web seulement)
# chaque pattern porte sa RELATION semantique : elle est indexee avec le
# triplet et permet la couverture multi-mots (question « capitale de
# l australie » active 'capitale' ET 'australie' = vote 2/2)
_PATTERNS_TRIPLETS = (
    (re.compile(r"^(?P<s>[A-ZÉÈ][\wéèêàçù'\- ]{2,40}?)\s+est\s+la\s+capitale\s+de\s+(?:la\s+|le\s+|l')?(?P<o>[\wéèêàçù'\- ]{2,40})[.\s]*$", re.IGNORECASE),
     "capitale de"),
    (re.compile(r"^(?P<s>[A-ZÉÈ][\wéèêàçù'\- ]{2,40}?)\s+est\s+le\s+president\s+de\s+(?:la\s+|le\s+|l')?(?P<o>[\wéèêàçù'\- ]{2,40})[.\s]*$", re.IGNORECASE),
     "president de"),
    (re.compile(r"^(?P<s>[\wéèêàçù'\- ]{2,40}?)\s+se\s+trouve\s+dans\s+(?:la\s+|le\s+|l')?(?P<o>[\wéèêàçù'\- ]{2,40})[.\s]*$", re.IGNORECASE),
     "se trouve dans"),
    (re.compile(r"^(?P<s>[\wéèêàçù'\- ]{2,40}?)\s+a\s+ete\s+invente\s+en\s+(?P<o>\d{4})[.\s]*$", re.IGNORECASE),
     "invente en"),
)


def _normaliser(texte: str) -> list[str]:
    t = unicodedata.normalize("NFKD", texte.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return [m for m in _MOTS.findall(t) if m not in _STOPWORDS]


def _charger():
    global _chargé, _triplets, _index
    if _chargé:
        return
    _chargé = True
    if not os.path.exists(_FICHIER):
        return
    try:
        with open(_FICHIER, encoding="utf-8") as f:
            for i, ligne in enumerate(f):
                ligne = ligne.strip()
                if not ligne:
                    continue
                try:
                    t = json.loads(ligne)
                except json.JSONDecodeError:
                    continue
                _triplets.append(t)
                for mot in set(_normaliser(f"{t.get('s','')} {t.get('r','')} {t.get('o','')}")):
                    _index.setdefault(mot, set()).add(i)
    except OSError as e:
        LOG.warning("graphe illisible : %s", e)


def _sauver(t: dict) -> None:
    try:
        with open(_FICHIER, "a", encoding="utf-8") as f:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    except OSError as e:
        LOG.warning("graphe non ecrit : %s", e)


def ajouter(sujet: str, relation: str, objet: str, source: str = "manuel") -> bool:
    """Ajoute un triplet (deduplique exact)."""
    s, r, o = sujet.strip(), relation.strip().lower(), objet.strip()
    if not s or not r or not o:
        return False
    with _verrou:
        _charger()
        cle = (s.lower(), r, o.lower())
        for t in _triplets:
            if (t["s"].lower(), t["r"], t["o"].lower()) == cle:
                return False
        i = len(_triplets)
        t = {"s": s, "r": r, "o": o, "src": source,
             "t": time.strftime("%Y-%m-%d")}
        _triplets.append(t)
        for mot in set(_normaliser(f"{s} {r} {o}")):
            _index.setdefault(mot, set()).add(i)
        _sauver(t)
    LOG.info("[graphe] triplet ajoute : %s -%s-> %s", s, r, o)
    return True


def extraire_de_reponse(reponse: str) -> list[tuple[str, str, str]]:
    """Extrait [(sujet, relation, objet)] d'une reponse (conservateur)."""
    trouves = []
    for ligne in reponse.split("."):
        ligne = ligne.strip()
        if not ligne:
            continue
        for pat, relation in _PATTERNS_TRIPLETS:
            m = pat.match(ligne)
            if m:
                trouves.append((m.group("s").strip(), relation,
                                m.group("o").strip()))
                break
    return trouves


def apprendre_de_reponse(reponse: str, verifiee_web: bool = False) -> int:
    """Nourrit le graphe depuis une reponse VERIFIEE web seulement.

    Renvoie le nombre de triplets effectivement ajoutes. Sans verification
    web, on n'apprend rien : le graphe ne doit jamais contenir d'erreur
    (c'est lui qu'on consulte pour eviter les hallucinations).
    """
    if not verifiee_web:
        return 0
    n = 0
    for s, r, o in extraire_de_reponse(reponse):
        if ajouter(s, r, o, source="web_verifie"):
            n += 1
    return n


def chercher(question: str, max_resultats: int = 5) -> str:
    """Contexte = triplets couvrant le plus de mots de la question.

    Multi-mots = multi-hop naturel. Vide = rien de pertinent (le modele
    repondra de lui-meme, comme avant).
    """
    _charger()
    if not _triplets:
        return ""
    mots = set(_normaliser(question))
    if len(mots) < 2:
        return ""
    scores: dict[int, int] = {}
    for m in mots:
        for i in _index.get(m, ()):
            scores[i] = scores.get(i, 0) + 1
    if not scores:
        return ""
    # au moins la moitie des mots couverts (hors stopwords) OU 2 mots minimum
    tri = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    lignes = []
    for i, couverture in tri[:max_resultats]:
        if couverture < 2:
            break
        t = _triplets[i]
        lignes.append(f"- {t['s']} : {t['r']} {t['o']}")
    return "\n".join(lignes)


def vider() -> None:
    """Reinitialise le graphe (fichier inclus)."""
    global _chargé, _triplets, _index
    with _verrou:
        _chargé = False
        _triplets, _index = [], {}
        try:
            if os.path.exists(_FICHIER):
                os.remove(_FICHIER)
        except OSError as e:
            LOG.warning("graphe non supprime : %s", e)


def stats() -> dict:
    _charger()
    return {"triplets": len(_triplets), "fichier": _FICHIER}
