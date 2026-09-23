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

# comparaisons textuelles -> (superieur, inferieur)
_RE_PLUS = re.compile(
    r"(\w+)\s+est\s+(?:plus\s+|le\s+)?(age|grand|grande|vieux|vieille|"
    r"lourd|gros|rapide|fort|cher|haute|haut|long|petit|jeune|leger)"
    r"e?\s+que\s+(\w+)", re.IGNORECASE)
_RE_MOINS = re.compile(
    r"(\w+)\s+est\s+(?:moins\s+)?(age|grand|grande|vieux|vieille|"
    r"lourd|gros|rapide|fort|cher|haute|haut|long)\s+que\s+(\w+)",
    re.IGNORECASE)
# signes bruts : A > B ou A < B
_RE_SIGNE = re.compile(r"(\w+)\s*(>)\s*(\w+)|(\w+)\s*(<)\s*(\w+)")
# mots interrogatifs de recherche d'extremum
_RE_MAX = re.compile(r"\b(le plus (?:age|grand|vieux|lourd|rapide|fort)|"
                     r"qui est le plus|le maximum|le plus age)\b", re.I)
_RE_MIN = re.compile(r"\b(le plus (?:petit|jeune|leger)|le minimum)\b", re.I)
_RE_ORDRE = re.compile(r"\b(dans l'ordre|du plus .* au plus .*\bordre)\b",
                       re.I)

_MOTS_OUT = {"que", "et", "puis", "ensuite", "de", "du", "le", "la", "les",
             "un", "une", "qui", "il", "elle", "on", "moi", "toi"}


def _est_entite(mot: str) -> bool:
    return bool(mot and mot.lower() not in _MOTS_OUT and not mot.isdigit()
                and len(mot) > 1)


def extraire_relations(texte: str) -> list[tuple[str, str]]:
    """Extrait les paires (superieur, inferieur) du texte, ou []."""
    t = texte.replace("'", "' ")
    paires: list[tuple[str, str]] = []
    # « A est plus age que B » -> (A, B) ; « A est plus petit que B »
    # -> (B, A) (petit/jeune inversent le sens)
    for m in _RE_PLUS.finditer(texte):
        a, adj, b = m.group(1), m.group(2).lower(), m.group(3)
        if not (_est_entite(a) and _est_entite(b)):
            continue
        inverse = adj in ("petit", "jeune", "leger")
        paires.append((b, a) if inverse else (a, b))
    # signes : A > B -> (A, B) ; A < B -> (B, A)
    for m in _RE_SIGNE.finditer(texte):
        if m.group(2):
            paires.append((m.group(1), m.group(3)))
        elif m.group(5):
            paires.append((m.group(6), m.group(4)))
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
