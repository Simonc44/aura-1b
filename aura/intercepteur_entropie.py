"""Intercepteur d'entropie : detecte l'hesitation du modele token par token.

Pour chaque token genere, on regarde la marge de probabilite entre le
token le plus probable (p1) et le second (p2). Une marge faible signifie
que le modele hesite : c'est le point ou un chemin deterministe (PAL,
solveur) peut prendre le relais.

Module PUR : aucune dependance au modele. Les logprobs viennent de
llama-cpp-python (`logprobs=True, top_logprobs=N`). Le seuil par defaut
n'est PAS calibre : il doit etre mesure sur le GGUF avant tout cablage.
"""
from __future__ import annotations

import math

SEUIL_MARGE = 0.15


def marge_top2(top_logprobs: dict[str, float]) -> float:
    """Ecart de probabilite p1 - p2 entre les deux meilleurs tokens.

    `top_logprobs` : {token: logprob}, comme renvoye par llama-cpp.
    Renvoie 1.0 s'il n'y a qu'un seul candidat (aucune hesitation possible),
    et 0.0 si la liste est vide (information absente : on suppose le doute).
    """
    if not top_logprobs:
        return 0.0
    probas = sorted((math.exp(lp) for lp in top_logprobs.values()), reverse=True)
    if len(probas) == 1:
        return 1.0
    return probas[0] - probas[1]


def premier_doute(marges: list[float], seuil: float = SEUIL_MARGE) -> int | None:
    """Index du premier token dont la marge passe sous `seuil`, sinon None."""
    for i, m in enumerate(marges):
        if m < seuil:
            return i
    return None
