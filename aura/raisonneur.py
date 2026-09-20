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
    "ETAPE 1 : <phrase courte de raisonnement> = <un seul calcul arithmetique>\n"
    "ETAPE 2 : <phrase courte> = <calcul>\n"
    "...\n"
    "REPONSE : <la reponse finale en une phrase>\n"
    "Regles :\n"
    "- chaque ETAPE contient EXACTEMENT un calcul avec des nombres\n"
    "- jamais plusieurs operations dans une meme etape si evitable\n"
    "- le calcul utilise des nombres, jamais des noms\n"
    "- la REPONSE finale s'appuie sur le dernier resultat"
)

_EXTRACT_CALCUL = re.compile(r"=\s*([-+]?\d+(?:\.\d+)?(?:\s*[-+*/]\s*[-+]?\d+(?:\.\d+)?)+)\s*$")
_EXTRACT_ETAPE = re.compile(r"ETAPE\s*\d+\s*:(.*)", re.IGNORECASE)


def extraire_etapes(texte: str) -> list[tuple[str, str]]:
    """Extrait [(phrase, calcul)] des lignes 'ETAPE n : ... = calcul'."""
    etapes = []
    for ligne in texte.splitlines():
        m = _EXTRACT_ETAPE.search(ligne)
        if not m:
            continue
        corps = m.group(1).strip()
        mc = _EXTRACT_CALCUL.search(corps)
        if mc:
            etapes.append((corps[:mc.start()].strip(), mc.group(1).strip()))
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
    lignes.append("Utilise ces valeurs EXACTES pour la reponse finale.")
    return "\n".join(lignes)
