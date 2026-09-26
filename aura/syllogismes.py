"""Syllogismes : inclusion et exclusion deduites regle par regle.

« tous les chats sont des felins, felix est un chat : felix est-il un
felin ? » ne rentre dans aucun expert : le tri transitif veut des
comparaisons chainees, le mini-SAT un formalisme entites/domaine que le
1B doit ecrire lui-meme. Ici tout est DETERMINISTE : lecture en jetons
des premisses (tous/tout, aucun, X est un Y), fermeture transitive de
l'inclusion, propagation des exclusions, puis reponse a la conclusion
demandee. Si la conclusion n'est ni derivable ni refutee, on s'abstient
(None) : cascade normale, jamais d'invention.
"""
from __future__ import annotations

import unicodedata

# sujets ou articles qui ne peuvent pas etre une entite
_STOP = frozenset((
    "ce", "cela", "cet", "cette", "il", "elle", "on", "qui", "que",
    "qu", "si", "et", "ou", "donc", "alors", "reponds", "oui", "non",
    "peut", "dire", "est", "un", "une", "des", "les", "le", "la", "n",
))


def _sing(mot: str) -> str:
    """Pluriel grossier -> singulier (felins -> felin), partout pareil."""
    return mot[:-1] if len(mot) > 3 and mot.endswith("s") else mot


def _jetons(texte: str) -> list[str]:
    """Normalise (bas de casse, sans accents ni ponctuation) en jetons."""
    n = (texte.lower().replace(chr(39), " ")
             .replace(chr(8217), " ").replace(chr(45), " "))
    nf = unicodedata.normalize("NFKD", n)
    n = "".join(c for c in nf if not unicodedata.combining(c))
    return [m for m in
            ("".join(ch for ch in mot if ch.isalnum()) for mot in n.split())
            if m]


def _couper(j: list[str]) -> tuple[list[str], list[str] | None]:
    """Separe premisses et question au declencheur interrogatif."""
    for i in range(len(j) - 2):
        if j[i:i + 3] == ["peut", "on", "dire"]:
            return j[:i], j[i:]
        if j[i:i + 3] == ["est", "ce", "que"]:
            return j[:i], j[i:]
        if j[i:i + 2] == ["est", "il"] or j[i:i + 2] == ["est", "elle"]:
            return j[:max(0, i - 1)], j[max(0, i - 1):]
    return j, None


def _regles(pre: list[str]) -> tuple[list, list, list]:
    """Extrait (inclusions, exclusions, instances) des premisses."""
    incl: list[tuple[str, str]] = []
    excl: list[tuple[str, str]] = []
    inst: list[tuple[str, str]] = []
    k = 0
    while k < len(pre):
        if pre[k] == "tous" and k + 5 < len(pre) and pre[k + 1] == "les"                 and pre[k + 3] == "sont" and pre[k + 4] == "des":
            incl.append((_sing(pre[k + 2]), _sing(pre[k + 5])))
            k += 6
        elif pre[k] == "tout" and k + 3 < len(pre) and pre[k + 2] == "est":
            incl.append((_sing(pre[k + 1]), _sing(pre[k + 3])))
            k += 4
        elif pre[k] == "aucun" and k + 4 < len(pre) and pre[k + 2] == "n"                 and pre[k + 3] == "est":
            m = k + 4
            if pre[m] in ("un", "une"):
                m += 1
            if m < len(pre):
                excl.append((_sing(pre[k + 1]), _sing(pre[m])))
            k = m + 1
        elif pre[k] == "est" and k + 2 < len(pre) and pre[k + 1] in ("un", "une")                 and k >= 1 and pre[k - 1] not in _STOP:
            inst.append((_sing(pre[k - 1]), _sing(pre[k + 2])))
            k += 3
        else:
            k += 1
    return incl, excl, inst


def _conclusion(q: list[str]) -> tuple[str, str, str] | None:
    """Lit la conclusion demandee apres « peut on dire que » ou « est il »."""
    base = 0
    if "dire" in q:
        base = q.index("dire") + 1
    while base < len(q) and q[base] in ("qu", "que"):
        base += 1
    for i in range(base, len(q)):
        if q[i] == "aucun" and i + 4 < len(q) and q[i + 2] == "n"                 and q[i + 3] == "est":
            m = i + 4
            if q[m] in ("un", "une"):
                m += 1
            if m < len(q):
                return ("exclut", _sing(q[i + 1]), _sing(q[m]))
        if q[i] == "tous" and i + 5 < len(q) and q[i + 1] == "les"                 and q[i + 3] == "sont" and q[i + 4] == "des":
            return ("inclut", _sing(q[i + 2]), _sing(q[i + 5]))
        if q[i] in ("est",) and i + 2 < len(q) and q[i + 1] in ("un", "une")                 and i >= 1 and q[i - 1] not in _STOP                 and q[i - 1] not in ("dire",):
            return ("membre", _sing(q[i - 1]), _sing(q[i + 2]))
        if q[i] in ("il", "elle") and i >= 2 and q[i - 1] == "est":
            m = i + 1
            if m < len(q) and q[m] in ("un", "une"):
                m = m + 1
            if m < len(q):
                return ("membre", _sing(q[i - 2]), _sing(q[m]))
    return None


def _supersets(incl: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Fermeture transitive de l'inclusion A -> {B inclus dans A}."""
    g: dict[str, set[str]] = {}
    for a, b in incl:
        g.setdefault(a, set()).add(b)
        g.setdefault(b, set())
    change = True
    while change:
        change = False
        for a in list(g):
            nouveaux = set()
            for b in g[a]:
                nouveaux |= g.get(b, set())
            if not nouveaux <= g[a]:
                g[a] |= nouveaux
                change = True
    return g


def repondre(question: str) -> str | None:
    """Oui / Non si la conclusion demandee se deduit des premisses.

    Aucune invention : conclusion absente ou non derivable -> None.
    """
    j = _jetons(question)
    if len(j) < 5:
        return None
    pre, q = _couper(j)
    if not q:
        return None
    incl, excl, inst = _regles(pre)
    if not incl and not excl and not inst:
        return None
    conc = _conclusion(q)
    if conc is None:
        return None
    sup = _supersets(incl)

    if conc[0] == "membre":
        _, x, c = conc
        for xx, b in inst:
            if xx != x:
                continue
            if c == b or c in sup.get(b, set()):
                return "Oui."
            for d in ({b} | sup.get(b, set())):
                if (d, c) in excl:
                    return "Non."
        return None

    if conc[0] == "inclut":
        _, a, c = conc
        if c in sup.get(a, set()):
            return "Oui."
        dom = {a} | sup.get(a, set())
        if any((d, c) in excl for d in dom):
            return "Non."
        return None

    # exclut
    _, a, c = conc
    if (a, c) in excl or any((d, c) in excl for d in sup.get(a, set())):
        return "Oui."
    if c in sup.get(a, set()):
        return "Non."
    return None
