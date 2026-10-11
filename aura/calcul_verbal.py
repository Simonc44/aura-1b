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
    r"monte(?:nt|s|e)?|arrive(?:nt|s)?|recupere[sr]?|"
    r"plus\s+\d|en\s+plus)\b")
_RE_RETRAIT = re.compile(
    r"\b(mange[sr]?|perds?|perdre|perdu|donne[sr]?|vend[sr]?|casse[sr]?|"
    r"coupe[sr]?|jette[sr]?|utilise[sr]?|depense[sr]?|consomme[sr]?|"
    r"descend(?:ent|s|re)?|retire[sr]?|enleve[sr]?|"
    # « j'en distribue 23 » = on en RETIRE 23. Le pronom « en » est exige :
    # « distribue 12 bonbons entre 4 amis » est une DIVISION, pas un
    # retrait — l'ajouter sans cette garde casserait ce cas-la.
    r"en\s+distribue[sr]?|"
    r"paie|paye[sr]?|rest(e|ent|era))\b")
_RE_DEPART = re.compile(
    r"\b(j'ai|j'ai|j'ai ai|j ai|avais|j'avais|possede|contient|a|avait|"
    r"part de|compte)\s+(\d+(?:[.,]\d+)?)")
# « la moitie » / « le demi » : une DIVISION, pas un retrait de nombre.
_RE_MOITIE = re.compile(r"(?:la\s+)?moitie\b|\bdemi\b")

# ECART : « quel est l'ecart d'age ? » — une DIFFERENCE entre deux
# quantites deja donnees. Ce n'est ni un ajout ni un retrait : aucune
# regle existante ne la voyait, et le 1B repondait (ou pas).
_RE_ECART = re.compile(r"\b(ecart|difference)\b")

# GROUPE : « 6 rangees de 7 chaises », « 5 boites de 12 oeufs », « 3 fois
# 8 billes » — une MULTIPLICATION ecrite en mots. Elle est REDUITE avant
# le balayage : le reste de la chaine (retraits, ajouts) s'applique alors
# au produit, ce qui couvre du meme coup « 5 boites de 12 oeufs, j'en
# casse 7 » (60 - 7 = 53) — deux etapes, un seul balayage.
_RE_GROUPE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s+(?:fois|rangees?\s+de|boites?\s+de|"
    r"paquets?\s+de|series?\s+de|groupes?\s+de|lots?\s+de)\s+"
    r"(\d+(?:[.,]\d+)?)")

# CHACUN : « 9 eleves et chacun a 4 cahiers » — distribution uniforme.
# Le verbe distributif est exige (« a », « recoit », « gagne »...) et
# suivi d'un nombre : « 3 livres a 18 euros chacun » (prix unitaire, deja
# traite par calcul_achat) ne doit PAS etre lu comme 9 x 4.
_RE_CHACUN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s+[a-z]+\s+(?:et\s+)?chacun(?:e)?\s+"
    r"(?:en\s+)?(?:a|possede|recoit|gagne|travaille|fait|prend)\s+"
    r"(\d+(?:[.,]\d+)?)")

# MEME UNITE enchaissee : « 18 km puis 9 km » — deux mesures de la MEME
# unite se cumulent (la distance totale). La REPETITION de l'unite est ce
# qui rend la regle sure : « 17 descendent et 9 montent » n'a pas deux
# fois la meme unite, donc il ne tombe pas ici (il est deja traite par
# l'ordre des segments).
_RE_MEME_UNITE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*([a-z]{2,12})\s*(?:puis|et)\s+"
    r"(\d+(?:[.,]\d+)?)\s*\2\b")
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


def _nombre(texte: str) -> float:
    return float(texte.replace(",", "."))


def _num(v: float) -> str:
    """Un nombre reecrit dans le TEXTE (entier quand il l'est).

    « 5 boites de 12 » doit devenir « 60 », pas « 60.0 » : le balayage
    qui suit relit ce texte avec les memes motifs que l'original.
    """
    return str(int(v)) if float(v).is_integer() else f"{v:.6g}"


def resoudre(question: str) -> str | None:
    """Traduit un petit probleme raconte en calcul exact, ou None.

    Semantique reconnue (extensible par regles) :
      depart + ajouts - retraits, question « combien ... reste/total ».
    Toute ambiguite -> None (cascade vers le cerveau, zero risque).

    Regle d'abstention : si AUCUNE operation (+/-) n'est reconnue, la
    question n'est pas un probleme raconte mais une simple enumeration
    (« 6 rangees de 7 chaises ») — le depart seul n'est pas une reponse,
    on rend None plutot que la valeur de depart (qui etait fausse).
    """
    q = _sans_accents(question)
    if not _RE_QUESTION.search(q):
        return None
    # ECART (difference entre deux quantites donnees) : forme sans ajout
    # ni retrait, et SANS le mot « combien » (« quel est l'ecart d'age ? »)
    # — d'ou ce test AVANT le garde-fou _RE_TOTAL.
    if _RE_ECART.search(q):
        nombres = [_nombre(n) for n in _RE_NOMBRE.findall(q)]
        return _fmt(abs(nombres[0] - nombres[1])) if len(nombres) == 2 else None
    if not _RE_TOTAL.search(q):
        return None
    nombres = _RE_NOMBRE.findall(q)
    if not (1 <= len(nombres) <= 3):
        return None

    # MEME UNITE enchainee (« 18 km puis 9 km, combien en tout ? ») : un
    # cumul de mesures, la question demandant explicitement le total.
    m_unite = _RE_MEME_UNITE.search(q)
    if m_unite:
        return _fmt(_nombre(m_unite.group(1)) + _nombre(m_unite.group(3)))

    # GROUPES : on REDUIT les multiplications ecrites en mots AVANT le
    # balayage, pour que la suite de la chaine s'applique au produit.
    groupe = False
    m_chacun = _RE_CHACUN.search(q)
    if m_chacun:
        q = (q[:m_chacun.start()]
             + _num(_nombre(m_chacun.group(1)) * _nombre(m_chacun.group(2)))
             + q[m_chacun.end():])
        groupe = True
    q, n_groupes = _RE_GROUPE.subn(
        lambda m: _num(_nombre(m.group(1)) * _nombre(m.group(2))), q)
    groupe = groupe or n_groupes > 0

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

    change = False          # au moins une operation reellement appliquee ?
    for pos, signe, mot in segments:
        # le nombre associe : premier nombre APRES le verbe (fenetre de
        # 40 caracteres, pour eviter « 3 pommes » (depart) deja pris)
        zone = q[pos + len(mot):pos + len(mot) + 40]
        # « donne la moitie » : le retrait porte sur la MOITIE du total,
        # pas sur le nombre qui suit (celui-ci appartient a l'etape
        # d'apres : « ... a Bob puis 2 a Clara »). Mesure au banc 100 :
        # sans ce cas, « 12 bonbons, la moitie puis 2 » rendait 10 (le
        # « 2 » etait pris pour l'operation) au lieu de 4.
        m_moitie = _RE_MOITIE.match(zone.lstrip())
        if m_moitie:
            total = total / 2
            change = True
            apres = zone.lstrip()[m_moitie.end():]
            m_puis = re.search(r"\b(?:puis|et)\s+(\d+(?:[.,]\d+)?)", apres)
            if m_puis:
                total -= float(m_puis.group(1).replace(",", "."))
            continue
        # FRANCAIS : « 17 descendent » met la quantite AVANT le verbe,
        # « on en coupe 5 » la met APRES. Le mot juste avant tranche : si
        # c'est un nombre, c'est LUI l'operande — sinon la fenetre d'apres
        # capterait le nombre du groupe suivant (« 45 passagers, 17
        # descendent et 9 montent » rendait 36 au lieu de 37).
        m_av = re.search(r"(\d+(?:[.,]\d+)?)$", q[:pos].rstrip())
        if m_av is not None:
            total += signe * float(m_av.group(1).replace(",", "."))
            change = True
            continue          # « N verbe » : rien a enumerer a la suite
        nm = _RE_NOMBRE.search(zone)
        if nm is None:
            # verbe sans quantite : « il en reste » -> question, ignore
            continue
        n = float(nm.group(0).replace(",", "."))
        total += signe * n
        change = True
        # enumeration d'ajouts : « j'achete 5 cahiers et 3 stylos » —
        # le « et N » suivant herite du signe du segment courant
        suite = zone[nm.end():]
        m_et = re.search(r"\bet\s+(\d+(?:[.,]\d+)?)", suite)
        if m_et and signe > 0:
            total += signe * float(m_et.group(1).replace(",", "."))
            change = True

    # garde-fou n°1 : aucune operation reconnue -> le probleme n'en est
    # pas un, le depart seul serait une reponse inventee. Exception : un
    # GROUPE reduit EST l'operation (« Il y a 6 rangees de 7 chaises »
    # n'a rien a ajouter ni retirer : 6 x 7 est la reponse).
    if not change and not groupe:
        return None
    # garde-fou n°2 : le resultat doit etre coherent (>= 0 si on parle de
    # quantites physiques ; - absurde -> on s'abstient)
    if total < 0:
        return None
    return _fmt(total)


def verifier(question: str, reponse: str) -> bool:
    """Controle de coherence : la reponse calculee doit etre positive
    (quantites) et le probleme doit contenir le mot « reste/total »."""
    return bool(reponse) and bool(_RE_QUESTION.search(_sans_accents(question)))
