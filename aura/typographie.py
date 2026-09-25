"""Typographie : le post-processeur C-level — la langue nettoyée par règles.

Le 1B écrit « le arbre », « reponse ?» sans espace, «  » au lieu de « ».
Ces corrections sont de l'ORDRE, pas du sens : elles appartiennent à des
regex précompilées (< 0,1 ms), jamais au LLM. Idempotent (relancer ne
change rien), et jamais appliqué à l'intérieur d'un bloc de code
(extraire les ``` de la réponse d'abord — voir executer_detaille).

Élision : « le arbre » -> « l'arbre ». La règle générale (voyelle ou h
minuscule = élision) est complétée par les EXCEPTIONS de la base lang.db
(h aspiré : « le héros » reste « le héros ») — consultation < 0,2 ms.
"""
from __future__ import annotations

import re

from .fast_lang import moteur as _lang

# « le/la » + mot à initiale voyelle ou h minuscule -> l' + mot.
# [^\w'] pour borner le mot sans avaler les apostrophes (l'arbre) ni
# coller à la ponctuation ; (\w[\w-]*) capture le mot suivant.
_RE_ELISION = re.compile(
    r"\b([Ll]e|[Ll]a)\s+([a-zà-ÿœA-ZÀ-ÿŒ][\wà-ÿœÀ-ÿŒ-]*)", re.UNICODE)
# espaces insécables AVANT ? ! : ; »  et APRÈS «
_RE_PONC_AVANT = re.compile(r"\s*([?!;:»])")
_RE_PONC_APRES = re.compile(r"(«)\s*")
# apostrophe droite -> typographique (' -> ' ; U+2019)
_RE_APOSTROPHE = re.compile(r"(?<=[\wà-ÿ])'(?=[\wà-ÿ])")

_ELISION_REMAP = {"le": "l'", "la": "l'"}


def nettoyer(texte: str) -> str:
    """Nettoie la typographie d'une réponse. Idempotent, < 0,1 ms."""
    if not texte:
        return texte
    texte = _RE_ELISION.sub(_corriger_elision, texte)
    texte = _RE_PONC_AVANT.sub("\u00A0\\1", texte)
    texte = _RE_PONC_APRES.sub("\\1\u00A0", texte)
    texte = _RE_APOSTROPHE.sub("\u2019", texte)
    return texte


def _corriger_elision(m: re.Match) -> str:
    det = m.group(1)
    mot = m.group(2)
    verdict = _lang.elision(det, mot)
    if verdict == "l'":
        return (_ELISION_REMAP[det.lower()] if det.lower() in _ELISION_REMAP
                else det) + mot
    if verdict in ("le", "la"):       # h aspiré : on GARDE le/la
        return f"{det} {mot}"
    # verdict None (base absente) : règle générale conservatrice
    if mot[0].isupper():
        return f"{det} {mot}"         # nom propre : pas d'élision
    if re.match(r"^[aeiouyàâéèêîôûùœ]", mot.lower()):
        return ("l'" if det.lower() == "le" else "l'") + mot
    if mot[0].lower() == "h":
        return ("l'" if det.lower() == "le" else "l'") + mot
    return f"{det} {mot}"


def hors_code(texte: str) -> str:
    """Nettoie la prose mais JAMAIS l'intérieur des blocs ```code```
    (le code verifié doit partir tel quel — un espace insécable dans
    du Python le casserait). Les segments hors code sont traités.
    """
    if not texte or "```" not in texte:
        return nettoyer(texte)
    morceaux = texte.split("```")
    # les segments d'indice impair sont DANS un bloc de code
    for i in range(0, len(morceaux), 2):
        morceaux[i] = nettoyer(morceaux[i])
    return "```".join(morceaux)


if __name__ == "__main__":
    exemples = [
        "le arbre est grand, la école aussi!",
        "le héros traverse le hall et le hameau ?",
        "voici la reponse : 42!",
    ]
    for e in exemples:
        print(f"  {e!r}\n-> {nettoyer(e)!r}\n")
