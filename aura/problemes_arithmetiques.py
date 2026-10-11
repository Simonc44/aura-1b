"""Problemes a DEUX etapes que les solveurs a une etape ne voient pas.

Ce module est ne d'une mesure, pas d'une intuition. Le sous-banc de
provenance (`scripts/sous_banc.py`, 100 questions en ~4 s) liste les
questions qui tombent encore sur le LLM ; parmi celles qui sont du CALCUL
pur, il restait deux familles que rien ne couvrait :

  * **fractions ecrites en mots** — « Un reservoir de 60 litres est
    rempli aux deux tiers. Combien de litres contient-il ? » (attendu 40).
    Ni `filtre_instantane` (aucune expression a evaluer), ni
    `calcul_verbal` (aucun ajout ni retrait : une seule quantite), ni
    `calcul_taux` (aucun taux) ne la voient ;
  * **geometrie elementaire** — « Un rectangle mesure 7 cm de long et
    4 cm de large. Quel est son perimetre ? » (22) / « Quelle est son
    aire ? » (28).

Ces deux familles ont exactement le meme profil que les autres solveurs :
une forme fixe, des nombres, un resultat exact — et 40 s de generation
evitees a chaque fois.

Deux disciplines, les memes qu'ailleurs
---------------------------------------
1. **Arithmetique EXACTE.** Les fractions sont calculees avec
   `fractions.Fraction`, pas en flottant : « aux deux tiers de 60 »
   donne exactement 40, la ou 60 * (2/3) donne 39,99999999999999 et
   obligerait a arrondir un resultat qui devait etre juste.
2. **Abstention des qu'un element manque.** Deux unites differentes, une
   fraction sans quantite de depart, une forme geometrique inconnue :
   None -> cascade normale (le 1B). Jamais une reponse inventee par le
   traducteur.
"""
from __future__ import annotations

import re
import unicodedata
from fractions import Fraction

# Multiples ecrits en mots, et noms de fraction. « tiers » vaut 1/3 :
# « aux deux tiers » = 2 x 1/3.
_MULTIPLES: dict[str, int] = {
    "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7,
    "huit": 8, "neuf": 9, "dix": 10,
}
_FRACTIONS: dict[str, Fraction] = {
    "demi": Fraction(1, 2), "demie": Fraction(1, 2), "moitie": Fraction(1, 2),
    "tiers": Fraction(1, 3), "quart": Fraction(1, 4), "quarts": Fraction(1, 4),
    "cinquieme": Fraction(1, 5), "cinquiemes": Fraction(1, 5),
    "sixieme": Fraction(1, 6), "dixieme": Fraction(1, 10),
}

_RE_NOMBRE = re.compile(r"\d+(?:[.,]\d+)?")

# « aux deux tiers », « au quart », « la moitie », « trois quarts » —
# l'article est facultatif, le multiple aussi.
_RE_FRACTION = re.compile(
    r"\b(?:(?P<mult>" + "|".join(_MULTIPLES) + r")\s+)?"
    r"(?P<frac>" + "|".join(_FRACTIONS) + r")\b")

# Forme geometrique : deux dimensions dans la MEME unite.
_RE_RECT_DIM = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*([a-z]{1,6})?\s*de\s+long\s+et\s+"
    r"(\d+(?:[.,]\d+)?)\s*([a-z]{1,6})?\s*de\s+large")
_RE_PERIMETRE = re.compile(r"\bperimetre\b")
_RE_AIRE = re.compile(r"\baire\b")

_RE_QUESTION = re.compile(r"\b(combien|quel|quelle)\b")


def _sans_accents(texte: str) -> str:
    n = unicodedata.normalize("NFKD", texte.lower())
    n = "".join(c for c in n if not unicodedata.combining(c))
    return n.replace("\u2019", "'")


def _fmt(v: Fraction | float) -> str:
    """Un Fraction exact rendu lisible (entier si entier, 6 signes sinon)."""
    if isinstance(v, Fraction):
        if v.denominator == 1:
            return f"{v.numerator:,}".replace(",", " ")
        return f"{float(v):.6g}"
    if float(v).is_integer():
        return f"{int(v):,}".replace(",", " ")
    return f"{v:.6g}"


def _nombre(texte: str) -> Fraction:
    return Fraction(texte.replace(",", "."))


def _fraction(q: str) -> Fraction | None:
    """Fraction ecrite en mots (« aux deux tiers »), ou None."""
    m = _RE_FRACTION.search(q)
    if not m:
        return None
    frac = _FRACTIONS[m.group("frac")]
    mult = _MULTIPLES.get(m.group("mult") or "", 1)
    return frac * mult


def _geometrie(q: str) -> str | None:
    """Perimetre / aire d'un rectangle, ou None (forme ou unites inconnues)."""
    m = _RE_RECT_DIM.search(q)
    if not m:
        return None
    # Les dimensions doivent etre les seules quantites numeriques de
    # l'enonce; les autres nombres rendraient la demande ambigue.
    if len(_RE_NOMBRE.findall(q)) != 2:
        return None
    # Unites MELANGEES (« 7 cm de long et 4 m de large ») : la reponse
    # dependrait de l'unite choisie -> on s'abstient plutot que de
    # supposer. Une unite absente d'un cote est acceptee (elle est alors
    # celle de l'autre).
    u1, u2 = m.group(2) or "", m.group(4) or ""
    if u1 and u2 and u1 != u2:
        return None
    cote_long, cote_large = _nombre(m.group(1)), _nombre(m.group(3))
    if _RE_PERIMETRE.search(q):
        return _fmt(2 * (cote_long + cote_large))
    if _RE_AIRE.search(q):
        return _fmt(cote_long * cote_large)
    return None


def resoudre(question: str) -> str | None:
    """Resout une fraction en mots ou une figure simple, sinon None."""
    q = _sans_accents(question)
    if not _RE_QUESTION.search(q):
        return None

    # 1. GEOMETRIE (avant les fractions : « aux deux tiers d'un aire »
    #    n'existe pas, mais l'ordre rend les traces lisibles).
    rep = _geometrie(q)
    if rep is not None:
        return rep

    # 2. FRACTION : il faut UNE quantite de depart et UNE seule. Deux
    #    nombres = deux etapes -> c'est le terrain de calcul_verbal (qui
    #    traite deja « 12 bonbons, la moitie puis 2 »), pas le notre.
    frac = _fraction(q)
    if frac is None:
        return None
    nombres = _RE_NOMBRE.findall(q)
    if len(nombres) != 1:
        return None
    return _fmt(_nombre(nombres[0]) * frac)


def verifier(question: str, reponse: str) -> bool:
    """La reponse doit etre un nombre et la question poser une quantite."""
    return bool(reponse) and bool(_RE_QUESTION.search(_sans_accents(question)))
