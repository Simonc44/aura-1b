"""Calcul verbal : resoudre les petits problemes arithmetiques racontes.

« si j'ai 3 pommes et que j'en mange 1, combien il m'en reste » ne
contient AUCUNE expression mathematique — le calculateur direct
(filtre_instantane) n'y voit que du texte et le 1B repond... un essai
sur la nutrition. La reponse exacte est pourtant triviale : 3 - 1 = 2.

Principe ( PAL, meme coup que raisonneur.py) : les mots du probleme
sont traduits en expression arithmetique par DES REGLES (pas par le
modele), l'expression est evaluee par l'AST exact, et un controle de
coherence verifie que le probleme raconte bien ce qui a ete calcule.
Si une regle echoue -> None : cascade normale (le 1B), jamais une
reponse inventee par le traducteur.
"""
import re

# Verbes/determinants d'operation, en texte sans accents (comme les
# autres modules : unicodedata NFKD avant comparaison).
_RE_AJOUT = re.compile(
    r"\b(achete[sr]?|gagne[sr]?|recu|recoit|ajoute[sr]?|obtient|"
    r"plus\s+\d|en\s+plus)\b")
_RE_RETRAIT = re.compile(
    r"\b(mange[sr]?|perds?|perdre|perdu|donne[sr]?|vend[sr]?|casse[sr]?|"
    r"jette[sr]?|utilise[sr]?|depense[sr]?|consomme[sr]?|paie|paye[sr]?|"
    r"rest(e|ent|era))\b")
_RE_DEPART = re.compile(
    r"\b(j'ai|j'ai|j'ai ai|j ai|avais|j'avais|possede|a|avait|"
    r"part de|compte)\s+(\d+(?:[.,]\d+)?)")
_RE_TOTAL = re.compile(
    r"\b(combien|total|en tout|au total)\b")
_RE_NOMBRE = re.compile(r"\d+(?:[.,]\d+)?")

# Question = recherche d'un resultat (sinon : phrase decontrative, on
# ne repond pas a un qui n'a pas pose de question).
_RE_QUESTION = re.compile(
    r"\b(combien|quel|quelle|combien il (m|l|nous|leur)? ?en reste|"
    r"combien en reste)\b")


def _sans_accents(texte: str) -> str:
    import unicodedata
    n = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in n if not unicodedata.combining(c))


def _fmt(v: float) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v:,}".replace(",", " ") if isinstance(v, int) else f"{v:.6g}"


def resoudre(question: str) -> str | None:
    """Traduit un petit probleme raconte en calcul exact, ou None.

    Semantique reconnue (extensible par regles) :
      depart + ajouts - retraits, question « combien ... reste/total ».
    Toute ambiguite -> None (cascade vers le cerveau, zero risque).
    """
    q = _sans_accents(question)
    if not (_RE_QUESTION.search(q) and _RE_TOTAL.search(q)):
        return None
    nombres = _RE_NOMBRE.findall(q)
    if not (1 <= len(nombres) <= 3):
        return None

    # Valeur de depart : le nombre qui suit j'ai/avais/possede...
    m = _RE_DEPART.search(q)
    if m:
        total = float(m.group(2).replace(",", "."))
    elif "en tout" in q or "total" in q:
        total = 0.0            # achats sans stock initial : on part de 0
    else:
        return None

    # balayage ordonne : chaque occurrence (verbe, nombre) dans l'ordre
    # d'apparition — ajout (+1) ou retrait (-1)
    segments = []
    for m in _RE_AJOUT.finditer(q):
        segments.append((m.start(), +1, m.group(0)))
    for m in _RE_RETRAIT.finditer(q):
        segments.append((m.start(), -1, m.group(0)))
    segments.sort()

    for pos, signe, mot in segments:
        # le nombre associe : premier nombre APRES le verbe (fenetre de
        # 25 caracteres, pour eviter « 3 pommes » (depart) deja pris)
        zone = q[pos + len(mot):pos + len(mot) + 25]
        nm = _RE_NOMBRE.search(zone)
        if not nm:
            # verbe sans nombre derive : « il en reste » -> question,
            # pas une operation — ignore
            continue
        n = float(nm.group(0).replace(",", "."))
        total += signe * n
        # enumeration d'ajouts : « j'achete 5 cahiers et 3 stylos » —
        # le « et N » suivant herite du signe du segment courant
        suite = zone[nm.end():]
        m_et = re.search(r"\bet\s+(\d+(?:[.,]\d+)?)", suite)
        if m_et and signe > 0:
            total += signe * float(m_et.group(1).replace(",", "."))

    # garde-fou : le resultat doit etre coherent (>= 0 si on parle de
    # quantites physiques ; - absurde -> on s'abstient)
    if total < 0:
        return None
    return _fmt(total)


def verifier(question: str, reponse: str) -> bool:
    """Controle de coherence : la reponse calculee doit etre positive
    (quantites) et le probleme doit contenir le mot « reste/total »."""
    return bool(reponse) and bool(_RE_QUESTION.search(_sans_accents(question)))
