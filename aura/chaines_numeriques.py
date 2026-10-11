"""Chainages numeriques : resoudre par le CALCUL les enigmes a ecarts chiffres.

Defaut reel (banc 100, categorie logique) : 4 questions sur 8 tombent au LLM
alors qu'elles sont de l'algebre elementaire. Le 1B, lui, ecrit une chaine
fluente mais fausse — le mode d'echec documente des petits modeles
(« imitation, pas logique »).

Deux familles, traitees EXACTEMENT :

1. ANCRAGE + ECARTS
   « Alice a 3 ans de plus que Bob. Bob a 2 ans de plus que Clara. Clara a
     4 ans. Quel age a Alice ? »
   Chaque relation donne  valeur(X) - valeur(Y) = +-N ; chaque « X a N ans »
   ANCRE une valeur. On propage par substitution : aucune approximation,
   aucune generation.  -> 9 (le 1B annonce « la formule » sans calculer)

2. EXTREMUM SUR VALEURS EXPLICITES
   « Ana 25 ans, Ben 31 ans, Chloe 28 ans. Qui est le plus jeune ? »
   Les valeurs sont DONNEES : il suffit de les trier.  -> Chloe

ABSTENTION PAR DEFAUT. Le module rend None des qu'il ne peut pas conclure
exactement : valeur manquante, unites melangees, entite ambigue. Un solveur
qui devine serait pire que pas de solveur — c'est la cascade qui reprend.
"""
from __future__ import annotations

import re

# Unites acceptees. Volontairement restreintes : une unite inconnue fait
# s'abstenir (mieux vaut None qu'une conversion inventee).
_UNITE = (r"ans?|annees?|mois|jours?|heures?|minutes?|secondes?|euros?|"
          r"centimes?|dollars?|kg|kilos?|grammes?|cm|mm|metres?|km|"
          r"litres?|points?|billes?|bonbons?|cartes?|places?")
_ART = r"(?:le\s+|la\s+|les\s+|l'|un\s+|une\s+|des\s+)?"

# « X a N ans » — un ANCRAGE. Le lookahead est INDISPENSABLE : sans lui,
# « Alice a 3 ans de plus que Bob » ancrerait Alice=3, et la propagation
# partirait d'une valeur fausse (le genre d'erreur silencieuse qui rend un
# solveur exact plus dangereux qu'aucun solveur).
_RE_ANCRE = re.compile(
    r"(\w+)\s+a\s+(\d+)\s*(?:" + _UNITE + r")\b(?!\s+de\s+(?:plus|moins))",
    re.IGNORECASE)
# « Alice a 3 ans de plus que Bob » -> (Alice, 3, plus, Bob)
_RE_ECART = re.compile(
    r"(\w+)\s+a\s+(\d+)\s*(?:" + _UNITE + r")\s+de\s+(plus|moins)\s+que\s+"
    + _ART + r"(\w+)", re.IGNORECASE)
# « Ana 25 ans » (LISTE de valeurs, sans « a ») : « Clara a 4 ans » donne ici
# (« a », 4), rejete par la longueur du nom — l'ancre vient de _RE_ANCRE.
_RE_VALEUR = re.compile(r"(\w+)\s+(\d+)\s*(?:" + _UNITE + r")\b",
                        re.IGNORECASE)

# Toutes les unites citees dans la question, pour verifier la COHERENCE.
_RE_UNITE = re.compile(r"\b(" + _UNITE + r")\b", re.IGNORECASE)

_RE_MIN = re.compile(
    r"\b(le plus (?:petit|jeune|leger|lent|court)|le minimum|le moins|"
    r"le plus faible|la moins)\b", re.IGNORECASE)
_RE_MAX = re.compile(
    r"\b(le plus (?:age|vieux|grand|lourd|rapide|riche|cher)|le maximum|"
    r"le plus grand|le plus eleve)\b", re.IGNORECASE)


def extraire(question: str) -> tuple[dict[str, int], list[tuple[str, int, int, str]]]:
    """(valeurs connues, ecarts) ou un ecart est (a, delta signe, ?, b).

    L'ecart signifie : valeur(a) = valeur(b) + delta.
    """
    valeurs: dict[str, int] = {}
    # 1) les ANCRES d'abord : elles sont la verite de l'enonce (« Sam a
    #    10 ans »), les valeurs de liste ne doivent pas les ecraser.
    for nom, n in _RE_ANCRE.findall(question):
        if len(nom) > 1:
            valeurs[nom.lower()] = int(n)
    for nom, n in _RE_VALEUR.findall(question):
        if len(nom) > 1:
            valeurs.setdefault(nom.lower(), int(n))

    ecarts: list[tuple[str, int, int, str]] = []
    for a, n, sens, b in _RE_ECART.findall(question):
        if len(a) <= 1 or len(b) <= 1:
            continue
        delta = int(n) if sens.lower() == "plus" else -int(n)
        ecarts.append((a.lower(), delta, 0, b.lower()))
    return valeurs, ecarts


def propager(valeurs: dict[str, int],
             ecarts: list[tuple[str, int, int, str]]) -> dict[str, int]:
    """Complete `valeurs` par substitution, tant que ca progresse.

    Les chaines sont courtes (2-3 relations) : une passe par relation suffit,
    et la boucle bornee evite tout risque de non-terminaison.
    """
    for _ in range(len(ecarts) + 1):
        change = False
        for a, delta, _, b in ecarts:
            if b in valeurs and a not in valeurs:
                valeurs[a] = valeurs[b] + delta
                change = True
            elif a in valeurs and b not in valeurs:
                valeurs[b] = valeurs[a] - delta
                change = True
        if not change:
            break
    return valeurs


def _forme(nom: str, question: str) -> str:
    """Restaure la casse d'origine de `nom` telle qu'ecrite dans la question."""
    m = re.search(rf"\b({re.escape(nom)})\b", question, re.IGNORECASE)
    return m.group(1) if m else nom


def _unites_coherentes(question: str) -> bool:
    """Toutes les valeurs portent-elles la MEME grandeur ?

    « Alice a 3 kg de plus que Bob. Bob a 2 ans. » n'est pas une enigme
    coherente : propager melangerait une masse et une duree. Une unite
    melangee est une raison de S'ABSTENIR, pas de deviner.
    """
    unites = {u.lower().rstrip("s") for u in _RE_UNITE.findall(question)}
    return len(unites) <= 1


def resoudre(question: str) -> str | None:
    """Reponse exacte a une enigme a ecarts chiffres, ou None (abstention)."""
    if not _unites_coherentes(question):
        return None
    valeurs, ecarts = extraire(question)
    if not ecarts and len(valeurs) < 2:
        return None                       # rien a deduire : pas une enigme
    propager(valeurs, ecarts)

    # ── famille 2 : extremum sur valeurs EXPLICITES ───────────────────
    # On exige l'absence d'ecarts : sinon les valeurs ne sont pas toutes
    # donnees, et un maximum « calcule » sur un sous-ensemble serait faux.
    if not ecarts and len(valeurs) >= 2:
        veut_min = bool(_RE_MIN.search(question))
        veut_max = bool(_RE_MAX.search(question))
        if veut_min == veut_max:          # ni l'un ni l'autre, ou les deux
            return None
        cible = (min if veut_min else max)(valeurs, key=lambda e: valeurs[e])
        return _forme(cible, question)

    # ── famille 1 : valeur d'une entite NOMMEE ────────────────────────
    # On ne retient que les entites citees dans la partie INTERROGATIVE
    # (apres le dernier point) : « Quel age a Alice ? ». Citer une entite
    # dans l'enonce ne suffit pas a en faire la question.
    partie = question.rsplit(".", 1)[-1].lower()
    citees = [e for e in valeurs
              if re.search(rf"\b{re.escape(e)}\b", partie)]
    if len(citees) == 1:
        return str(valeurs[citees[0]])
    return None
