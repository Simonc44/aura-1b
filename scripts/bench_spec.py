"""Micro-bench du speculative decoding (prompt-lookup / n-gram).

Deux passes dans DEUX processus distincts (le modele est charge une fois) :

    .venv/Scripts/python.exe scripts/bench_spec.py 1   # spec ON  (defaut)
    AURA_SPEC=0 .venv/Scripts/python.exe scripts/bench_spec.py 0  # spec OFF

Le chiffre utile est celui du log ``[llama] N tokens en Xs = Y tok/s``
(passe target) : le draft n-gram ne coute rien sur du texte neuf ([]) et
gagne sur du texte repetitif (code, listes, tours de conversation).
"""
from __future__ import annotations

import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AURA_SPEC", "1")
    from aura import llama_cerveau as lc

    llm = lc._charger()
    draft = getattr(llm, "draft_model", None)
    print(f"[bench] AURA_SPEC={mode} draft_model={type(draft).__name__ if draft else None}")

    # 1) texte repetitif : le cas ou le n-gram draft peut gagner
    q_rep = ("Ecris 40 lignes numérotées du style "
             "'Ligne 4 : le chat dort sous la table' (phrase identique à chaque ligne).")
    # 2) texte neuf : cas nominal, le draft renvoie [] (pas de perte)
    q_neuf = ("Explique en 3 phrases pourquoi le ciel est bleu pendant la journée.")

    for nom, q, budget in (("repétitif", q_rep, 160), ("neuf", q_neuf, 160)):
        t0 = time.time()
        r = lc.generer(q, max_tokens=budget)
        dt = time.time() - t0
        print(f"[bench] {nom:10s} {dt:5.1f}s  {len(r)} car.")

    llm.reset() if hasattr(llm, "reset") else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
