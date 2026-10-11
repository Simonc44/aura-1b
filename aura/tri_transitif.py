"""Tri transitif : resoudre les comparaisons en chaine (A > B > C).

Les enigmes « Paul est plus age que Marie, Marie est plus age que Leo,
qui est le plus age ? » ne rentrent dans AUCUN expert existant :
- pas de maths (niveau 0 : pas d'expression),
- le mini-SAT (logique.py) attend un formalisme ENTITES/DOMAINE/CONDITION
  que le 1B doit ecrire — trop lourd pour une simple chaine d'ordre.

Ici tout est DETERMINISTE : regex de comparaison (plus age/grand/petit/
jeune, >, <) -> paires (superieur, inferieur) -> graphe oriente ->
tri topologique. Reponses derivees : maximum, minimum, ordre complet.
Aucun mot genere par le modele : si le parseur n'extrait rien de
solide, None -> cascade normale (zero risque d'invention).
"""
import re
from collections import defaultdict

from . import syllogismes

# ── comparaisons ──────────────────────────────────────────────────────
# Un SEUL motif couvre « plus » et « moins », tous les adjectifs, et le
# VERBE intercalaire. Trois defauts corriges le 6 oct. 2026, chacun ayant
# coute des questions du banc :
#   1. l'ARTICLE entre « que » et l'entite n'etait pas prevu : sur « le
#      chien est plus lourd que le chat », (\w+) capturait « le », la paire
#      etait jetee, et le solveur s'abstenait (logique-04/06) ;
#   2. seul le verbe « est » etait accepte, donc « coute plus cher qu'un
#      stylo » ne donnait rien (logique-06) ;
#   3. « moins » n'etait JAMAIS lu : _RE_MOINS existait mais n'etait appele
#      nulle part (motif mort).
_ADJ = (r"age(?:s)?|grand(?:e)?s?|vieux|vieille|vielle?|lourd(?:e)?s?|"
       r"gros(?:se)?s?|rapide(?:s)?|fort(?:e)?s?|cher|chere?|chere|haut(?:e)?s?|"
       r"long(?:ue)?s?|petit(?:e)?s?|jeune(?:s)?|leger|l[ée]g[èe]re?s?|"
       r"lent(?:e)?s?|riche(?:s)?|pauvre(?:s)?|intelligent(?:e)?s?|"
       r"court(?:e)?s?|drole|drole(?:s)?")
_ART = r"(?:le\s+|la\s+|les\s+|l'|un\s+|une\s+|des\s+)?"
# adjectifs dont le SENS est inverse : « plus petit que » = plus PETIT en
# valeur, donc X est SOUS Y. Comparaison sur la forme nue (sans e final).
_ADJ_INVERSES = frozenset(("petit", "jeune", "leger", "lent", "court",
                           "pauvre"))

_RE_COMP = re.compile(
    r"(\w+)\s+(?:\w+\s+)?(plus|moins)\s+(" + _ADJ + r")\s+"
    r"qu'?e?\s*" + _ART + r"(\w+)", re.IGNORECASE)
# signes bruts : A > B ou A < B
_RE_SIGNE = re.compile(r"(\w+)\s*(>)\s*(\w+)|(\w+)\s*(<)\s*(\w+)")
# mots interrogatifs de recherche d'extremum
_RE_MAX = re.compile(r"\b(le plus (?:age|grand|vieux|lourd|rapide|fort)|"
                     r"qui est le plus|le maximum|le plus age)\b", re.I)
# « devant » ET « avant » : les deux disent la meme chose dans une course
# (« A arrive avant B », logique-08), et l'article est optionnel.
_RE_DEVANT = re.compile(
    r"(\w+)\s+(?:est|arrive|termine|fini[sz]?)\s+(?:devant|avant)\s+"
    + _ART + r"(\w+)", re.I)
# « le MINIMUM » se dit aussi « coute le MOINS cher » : sans cette branche,
# logique-06 rendait l'ordre complet au lieu de repondre a la question.
_RE_MIN = re.compile(
    r"\b(le plus (?:petit|jeune|leger|lent|court|faible)|"
    r"le minimum|le moins(?:\s+\w+)?|dernier|la moins)\b", re.I)
_RE_ORDRE = re.compile(r"\b(dans l'ordre|du plus .* au plus .*\bordre)\b",
                       re.I)

_MOTS_OUT = {"que", "et", "puis", "ensuite", "de", "du", "le", "la", "les",
             "un", "une", "qui", "il", "elle", "on"}


def _est_entite(mot: str) -> bool:
    """Ce mot est-il une entite nommee (et pas un mot-outil) ?

    Les entites d'UNE lettre sont acceptees si elles sont MAJUSCULES :
    « A arrive avant B » (logique-08) parle bien de A et B, alors que
    « Clara a 4 ans » ne parle pas de l'entite « a ». Le filtre de mot-outil
    (que/et/le...) reste, lui, insensible a la casse.
    """
    if not mot or mot.lower() in _MOTS_OUT or mot.isdigit():
        return False
    return len(mot) > 1 or mot.isupper()


def _inverse(adj: str) -> bool:
    """L'adjectif inverse-t-il le sens ? (« plus petit que » -> sous Y)"""
    a = adj.lower()
    if a.endswith("e") and a[:-1] in _ADJ_INVERSES:
        a = a[:-1]
    return a in _ADJ_INVERSES


def extraire_relations(texte: str) -> list[tuple[str, str]]:
    """Extrait les paires (superieur, inferieur) du texte, ou []."""
    t = texte.replace("'", "' ")
    t = re.sub(r"\bje\s+suis\b", "moi est", t, flags=re.IGNORECASE)
    t = re.sub(r"\btu\ses\b", "toi est", t, flags=re.IGNORECASE)
    paires: list[tuple[str, str]] = []
    # « A est plus age que B » -> (A, B) ; « A est plus petit que B »
    # -> (B, A) ; « A est moins age que B » -> (B, A). Deux retournements
    # INDEPENDANTS : l'adjectif (petit/jeune/leger) et la polarite (moins).
    # Les combiner en XOR evite « moins petit » = plus grand.
    for m in _RE_COMP.finditer(t):
        a, polarite, adj, b = (m.group(1), m.group(2).lower(),
                               m.group(3).lower(), m.group(4))
        if not (_est_entite(a) and _est_entite(b)):
            continue
        inverse = _inverse(adj) != (polarite == "moins")
        paires.append((b, a) if inverse else (a, b))
    # signes : A > B -> (A, B) ; A < B -> (B, A)
    for m in _RE_SIGNE.finditer(texte):
        if m.group(2):
            paires.append((m.group(1), m.group(3)))
        elif m.group(5):
            paires.append((m.group(6), m.group(4)))
    for m in _RE_DEVANT.finditer(t):
        a, b2 = m.group(1), m.group(2)
        if _est_entite(a) and _est_entite(b2):
            paires.append((a, b2))
    return paires


def trier(paires: list[tuple[str, str]]) -> list[str] | None:
    """Tri topologique des entites (du plus grand au plus petit).

    Renvoie None si le graphe contient un cycle (contradiction) —
    on prefere s'abstenir que de trancher au hasard.
    """
    avant = defaultdict(set)          # x -> {ceux que x domine}
    for sup, inf in paires:
        avant[sup].add(inf)
        avant.setdefault(inf, set())
    # fermeture transitive par propagation (les chaines sont courtes)
    change = True
    while change:
        change = False
        for x in list(avant):
            nouveaux = set()
            for y in avant[x]:
                nouveaux |= avant[y]
            if not nouveaux <= avant[x]:
                avant[x] |= nouveaux
                change = True
    # cycle = x dans sa propre domination -> contradiction
    for x, domines in avant.items():
        if x in domines:
            return None
    # ordre : par nombre de dominés décroissant (le max en domine N-1)
    return sorted(avant, key=lambda x: -len(avant[x]))


def repondre(question: str) -> str | None:
    """Reponse exacte a une enigme de comparaison, ou None."""
    rep = syllogismes.repondre(question)
    if rep is not None:
        return rep
    paires = extraire_relations(question)
    if len(paires) < 2:               # une seule relation = pas une enigme
        return None
    ordre = trier(paires)
    if not ordre:
        return None
    q = question.lower()
    if _RE_MIN.search(q):
        return ordre[-1]
    if _RE_MAX.search(q) or "qui est le plus" in q:
        return ordre[0]
    if _RE_ORDRE.search(q):
        return " > ".join(ordre)
    # sans question explicite : chaine detectee -> l'ordre complet est
    # la reponse la plus utile (et toujours exacte)
    return " > ".join(ordre)
