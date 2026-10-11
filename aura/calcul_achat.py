"""Calcul d'achat : budget - somme des couts (prix unitaire x quantite).

Le 1B ne sait pas enchainer DEUX etapes arithmetiques sur un probleme
raconte. Mesure au banc 100 (scripts/banc_100.py) :

  « Un cahier coute 3 euros. J'en achete 4 et je paie avec un billet de
    20 euros. Combien me rend-on ? »
  -> il rendait « 12 euros + 20 euros = 32 euros » (il ADDITIONNE le
     billet au lieu de le soustraire) ; attendu 8.
  « J'ai 100 euros. J'achete 3 livres a 18 euros chacun. Combien me
    reste-t-il ? » -> il rendait 103 (100 + 3) ; attendu 46.

Ce n'est pas une limite de connaissance mais d'ENCHAINEMENT : deux
operations, un ordre. Le programme, lui, ne se trompe pas.

Une seule operation, deux formes reconnues :
  budget  = « j'ai N euros »                         (ce que je possede)
            « je paie avec un billet de N euros »     (ce que je donne)
  cout    = prix unitaire x quantite
            « <objet> coute N euros »  + « j'en achete M »
            « j'achete M <objets> a N euros (chacun) »
            « <objet> a N euros le <unite> » + « j'en achete M <unite> »
  reponse = budget - cout

Discipline identique a calcul_verbal : toute ambiguite -> None (cascade
vers le cerveau, zero risque). Un budget absent ou un cout introuvable
n'est PAS devine — on s'abstient.
"""
import re


def _sans_accents(texte: str) -> str:
    import unicodedata
    n = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in n if not unicodedata.combining(c))


def _fmt(v: float) -> str:
    if float(v).is_integer():
        return f"{int(v):,}".replace(",", " ")
    return f"{v:.6g}"


# Le probleme doit parler d'argent : sans cela, ce n'est pas un achat.
_RE_MONNAIE = re.compile(r"\b(?:euros?|centimes?|€)\b")

# Ce que je DONNE (paye) — prioritaire : c'est la valeur a laquelle on
# rend la monnaie.
_RE_BUDGET_PAIE = re.compile(
    r"\b(?:je\s+)?(?:paie|paye)\s+avec\s+(?:un\s+)?(?:billet\s+de\s+)?"
    r"(\d+(?:[.,]\d+)?)")
# Ce que je POSSEDE, si aucun paiement n'est mentionne.
_RE_BUDGET_AI = re.compile(
    r"\b(?:j'ai|j ai|je dispose de|je possede)\s+(\d+(?:[.,]\d+)?)")

# Prix unitaire, trois formulations.
_RE_PRIX_UNITE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:euros?|€)\s*(?:le|la|l'|par|/)\s*([a-z]+)")
_RE_COUTE = re.compile(
    r"\b(?:un|une|le|la|l')\s*([a-z]+)\s+coute\s+(\d+(?:[.,]\d+)?)\s*"
    r"(?:euros?|€)")
_RE_PRIX_SEUL = re.compile(
    r"\b(?:a|à)\s+(\d+(?:[.,]\d+)?)\s*(?:euros?|€)")

# ARTICLE chiffre : « 3 livres a 18 euros », « 2 cahiers a 5 euros ».
# findall() les somme tous : « 3 stylos a 2 euros et 2 cahiers a
# 5 euros » = 6 + 10 = 16 (mesure au banc 100 : le 1B rendait 173
# caracteres de bavardage au lieu de 16).
_RE_ARTICLE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s+[a-z]+\s+a\s+"
    r"(\d+(?:[.,]\d+)?)\s*(?:euros?|€)")
# « j'en achete 4 » / « j'en achete 3 kilos » : quantite seule.
_RE_QUANTITE = re.compile(r"\bj'en\s+achete\s+(\d+(?:[.,]\d+)?)")
# Question qui attend une MONNAIE RENDUE ou un RESTE.
_RE_QUESTION = re.compile(
    r"\b(combien|quel(?:le)?)\b|\breste\b|\brend\b|\brend-on\b")
# Intention de payer SANS budget annonce -> la question porte sur le
# TOTAL a payer, pas sur une monnaie a rendre.
_RE_INTENTION_PAIEMENT = re.compile(
    r"\b(?:je\s+)?(?:paie|paye)\b|\bca\s+fait\b|\bca\s+coute\b|"
    r"\bcoute(?:nt)?\b|\bprix\s+total\b")


def resoudre(question: str) -> str | None:
    """Rend la monnaie / le reste d'un achat, ou None si ambigu."""
    q = _sans_accents(question)
    if not _RE_MONNAIE.search(q) or not _RE_QUESTION.search(q):
        return None

    # -- budget (peut etre absent : question de total) -------------------
    m = _RE_BUDGET_PAIE.search(q) or _RE_BUDGET_AI.search(q)
    budget = float(m.group(1).replace(",", ".")) if m else None

    # -- cout : SOMME des articles chiffres ------------------------------
    articles = _RE_ARTICLE.findall(q)
    if articles:
        cout = sum(float(a.replace(",", ".")) * float(p.replace(",", "."))
                   for a, p in articles)
    else:
        for motif, groupe in ((_RE_PRIX_UNITE, 1), (_RE_COUTE, 2),
                             (_RE_PRIX_SEUL, 1)):
            m = motif.search(q)
            if m:
                prix = float(m.group(groupe).replace(",", "."))
                break
        else:
            return None
        mq = _RE_QUANTITE.search(q)
        qte = float(mq.group(1).replace(",", ".")) if mq else 1.0
        cout = prix * qte

    # -- pas de budget : la question porte sur le TOTAL a payer ----------
    if budget is None:
        if _RE_INTENTION_PAIEMENT.search(q):
            return _fmt(cout)
        return None

    # -- reste ----------------------------------------------------------
    reste = budget - cout
    if reste < 0:            # on ne peut pas rendre plus qu'on a donne
        return None
    return _fmt(reste)


def verifier(question: str, reponse: str) -> bool:
    """La reponse doit etre un nombre et la question parler d'argent."""
    return bool(reponse) and bool(_RE_MONNAIE.search(_sans_accents(question)))
