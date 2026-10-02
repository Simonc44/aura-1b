"""Raisonneur Program-of-Thoughts (PoT) : le 1B ecrit les etapes, l'AST verifie.

Inspiré de TIGER-AI-Lab/Program-of-Thoughts (TMLR 2023) : « desentrelacer
le calcul du raisonnement ». Le point faible des petits modeles n'est pas
la LOGIQUE d'un puzzle (« Paul a 3 ans de plus que Marie, Marie a le double
de Leo... ») mais l'arithmetique mentale en chaine. PoT inverse les roles :
le modele raisonne en etapes et ecrit un calcul, le programme l'execute
exactement (zero hallucination arithmetique).

Le puzzle est detecte par des indices lexicaux d'AGES/relations — c'est le
cas d'usage ou le 1B derape systematiquement et ou PoT rapporte le plus.
"""
import ast
import logging
import operator
import re
import unicodedata

LOG = logging.getLogger("aura.pot")

# -- evaluateur AST securise ( meme contrat que filtre_instantane ) ---------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}
_PUISSANCE_MAX = 1e15


def _evaluer(noeud):
    if isinstance(noeud, ast.Expression):
        return _evaluer(noeud.body)
    if isinstance(noeud, ast.Constant) and isinstance(noeud.value, (int, float)):
        return noeud.value
    if isinstance(noeud, ast.BinOp) and type(noeud.op) in _OPS:
        g, d = _evaluer(noeud.left), _evaluer(noeud.right)
        if isinstance(noeud.op, ast.Pow):
            if abs(d) > 64 or abs(g) ** min(abs(d), 64) > _PUISSANCE_MAX:
                raise ValueError("puissance hors limites")
        return _OPS[type(noeud.op)](g, d)
    if isinstance(noeud, ast.UnaryOp) and type(noeud.op) in _OPS:
        return _OPS[type(noeud.op)](_evaluer(noeud.operand))
    raise ValueError("expression non autorisee")


def _fmt(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v:,}".replace(",", " ") if isinstance(v, int) else f"{v:.6g}"


# -- detection de puzzle ------------------------------------------------------

# indices : ages relatifs, doubles/moities, sommes partagees — la famille de
# puzzles ou le 1B rate l'arithmetique mentale en chaine
_INDICES_PUZZLE = (
    "age", "ages", "ans de plus", "ans de moins", "le double", "la moitie",
    "fois plus", "de plus que", "de moins que", "ensemble ils",
    "a eux deux", "somme de leurs",
)
# une question de puzzle pose une question finale : « qui a ... », « quel age »
_MOT_QUESTION = re.compile(r"\b(quel age|qui a|combien d ans|combien a)\b")


def est_puzzle(question: str) -> bool:
    """Vrai si la question ressemble a un puzzle d'ages/relations."""
    q = unicodedata.normalize("NFKD", question.lower())
    q = "".join(c for c in q if not unicodedata.combining(c))
    if not _MOT_QUESTION.search(q):
        return False
    return any(indice in q for indice in _INDICES_PUZZLE)


# -- prompt PoT ----------------------------------------------------------------

_SYSTEME_POT = (
    "Tu resous des puzzles de logique par ETAPES NUMERIQUEES.\n"
    "Format OBLIGatoire :\n"
    "ETAPE 1 : <phrase courte> = <un calcul avec un operateur>\n"
    "ETAPE 2 : <phrase courte> = <calcul>\n"
    "...\n"
    "REPONSE : <la reponse finale en une phrase>\n"
    # EXEMPLE few-shot : sans modele concret a imiter, le 1B ecrit des
    # lignes libres (« Bob : 4 + 2 = 6 ans ») ou du algebrique
    # (« Alice = Bob + 3 ») que l'AST ne peut pas verifier -> PoT tombe a
    # 0 etape et laisse le puzzle au hasard. Banc de variantes mesure :
    # ce format donne 4/4 branches dont la derniere etape vaut la bonne
    # valeur (l'ancien n'en donnait 1/4, avec 23 et 8 comme reponses).
    "Exemple :\n"
    "ETAPE 1 : Marie a le double de Leo (5 ans) = 5 * 2\n"
    "ETAPE 2 : Paul a 3 ans de plus que Marie = 10 + 3\n"
    "REPONSE : Paul a 13 ans.\n"
    "Regles :\n"
    "- COMMENCE par les valeurs donnees dans l enonce, puis enchaine\n"
    "- 'X a n ans de plus que Y' se traduit par X = Y + n (jamais Y - n)\n"
    "- chaque calcul n utilise que des nombres DEJA connus\n"
    "- chaque ETAPE contient EXACTEMENT un calcul avec un operateur\n"
    "- jamais de noms dans le calcul, uniquement des nombres\n"
    "- la REPONSE finale s appuie sur le dernier resultat"
)

# Tolerant aux 3 formes REELLES du 1B (mesurees sur 10 echantillons) :
#   A canon   : ETAPE 1 : phrase = 4 + 4
#   B calc+res: ETAPE 1 : phrase = 4 + 4 = 8
#   C libre   : phrase = 6 + 3 = 9 ans      (sans prefixe ETAPE)
# L'AST (verifier_calculs) reste l'unique reference de verite : on EXTRAIT
# seulement ce qui ressemble a un calcul, on ne tranchera jamais ici.
_CALCUL_EXACT = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\s*[-+*/]\s*[-+]?\d+(?:\.\d+)?)+")
_NON_CALCUL = re.compile(r"[^0-9+\-*/.\s]")
_EXTRACT_ETAPE = re.compile(r"ETAPE\s*\d+\s*:(.*)", re.IGNORECASE)


def _calcul(fragment: str | None) -> str | None:
    """Normalise un fragment et renvoie son calcul exact, sinon None.

    Lettres et ponctuation sautent : « 6 ans » -> « 6 » (pas d'operateur,
    refuse), « Bob + 3 » -> « + 3 » (incomplet, refuse), tandis que
    « 4 + 4 » ressort intact de « 4 + 4 = 8 ».
    """
    if not fragment:
        return None
    propre = fragment.replace("\u00d7", "*").replace("\u00f7", "/")
    # le 1B ecrit « 2 x 4 » (lettre) pour la multiplication : chiffre autour
    # uniquement, pour ne pas avaler une variable « x »
    propre = re.sub(r"(?<=\d)\s*[xX]\s*(?=[-+]?\d)", " * ", propre)
    propre = _NON_CALCUL.sub(" ", propre)
    propre = re.sub(r"\s+", " ", propre).strip()
    return propre if _CALCUL_EXACT.fullmatch(propre) else None


def extraire_etapes(texte: str) -> list[tuple[str, str]]:
    """Extrait [(phrase, calcul)] des lignes ETAPE (formes A/B/C du 1B)."""
    etapes: list[tuple[str, str]] = []
    for ligne in texte.splitlines():
        m = _EXTRACT_ETAPE.search(ligne)
        corps = m.group(1).strip() if m else ligne.strip()
        if "=" not in corps:
            continue                    # prose sans calcul : aucune etape
        parties = [p.strip() for p in corps.split("=")]
        phrase = parties[0]
        if not phrase:
            continue
        # A/B/C : le calcul est a droite, ou incruste dans la phrase
        if len(parties) > 2:
            calcul = (_calcul(parties[-2]) or _calcul(parties[1])
                      or _calcul(phrase))
        else:
            calcul = _calcul(parties[1]) or _calcul(phrase)
        if calcul:
            etapes.append((phrase, calcul))
    return etapes


def verifier_calculs(etapes: list[tuple[str, str]]) -> list[float]:
    """Evalue chaque calcul d'etape avec l'AST. Leve si un calcul echoue."""
    resultats = []
    for _, calcul in etapes:
        try:
            arbre = ast.parse(calcul, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"calcul illisible : {calcul!r}") from e
        try:
            resultats.append(_evaluer(arbre))
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            raise ValueError(f"calcul refuse : {calcul!r} ({e})") from e
    return resultats


def construire_verification(etapes: list[tuple[str, str]],
                            resultats: list[float]) -> str:
    """Bloc injecte au modele : ses etapes AVEC les valeurs verifiees."""
    lignes = ["VERIFICATION DE TES ETAPES (calculees exactement) :"]
    for (phrase, calcul), res in zip(etapes, resultats):
        lignes.append(f"- {phrase} = {calcul} = {_fmt(res)}")
    # mesure : la synthese rederivait parfois malgre ce bloc (« 7 » au
    # lieu de « 9 ») — on interdit explicitement le recalcul maison
    lignes.append("Utilise ces valeurs EXACTES pour la reponse finale, "
                  "sans recalculer : reponds en UNE seule phrase.")
    return "\n".join(lignes)


def phrase_verifiee(etapes: list[tuple[str, str]],
                    resultats: list[float]) -> str:
    """Derniere etape AST-verifiee rendue en phrase, construite PAR LE CODE.

    Garde-fou de la synthese : si le modele ignore les valeurs verifiees,
    cette phrase (arithmetiquement exacte, l'AST l'a prouvee) complete la
    reponse — la bonne valeur ne disparait jamais de la sortie.
    """
    (phrase, calcul), res = etapes[-1], resultats[-1]
    return f"Verification exacte : {phrase} = {calcul} = {_fmt(res)}."
