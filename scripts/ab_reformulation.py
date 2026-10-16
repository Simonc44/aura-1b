"""A/B de la reformulation de requete (AURA_REFORMULATION) sur le reseau reel.

Pour chaque question, compare les sources retenues par la recherche web :
  A : la question brute ;
  B : la requete reformulee par le modele 1B (Orchestrateur._requete_web).
Les resultats sont ecrits en JSON pour un jugement manuel de pertinence.

Usage : python scripts/ab_reformulation.py --sortie ab.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUESTIONS = [
    "prix de l or aujourd'hui",
    "qui est le président de la France",
    "résultats du match de football hier soir",
    "cours du bitcoin maintenant",
    "dernière version de Python",
    "ouverture de la bibliothèque municipale de Lyon",
    "taux d'inflation en France 2026",
    "quand est la prochaine éclipse solaire visible en Europe",
    "meilleur framework JavaScript en 2026",
    "nouvelles de la NASA cette semaine",
    "date de sortie de la PS6",
    "coût d'un billet de train Paris Marseille",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sortie", default="ab_reformulation.json")
    args = parser.parse_args()

    from aura import memoire_web
    from aura.orchestrateur import RontoGPT

    os.environ["AURA_REFORMULATION"] = "1"
    sortie = []
    for q in QUESTIONS:
        reformulee = RontoGPT._requete_web(None, q)  # n'utilise pas self
        a = memoire_web.chercher(q, max_resultats=3)
        b = memoire_web.chercher(reformulee, max_resultats=3)
        sortie.append({"question": q, "reformulee": reformulee,
                       "sources_A": a.splitlines()[:6],
                       "sources_B": b.splitlines()[:6]})
        print(f"\n## {q}\n  -> {reformulee}")
        print("  A:", " | ".join(ligne[:90] for ligne in a.splitlines()[:3]))
        print("  B:", " | ".join(ligne[:90] for ligne in b.splitlines()[:3]))
    Path(args.sortie).write_text(json.dumps(sortie, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
