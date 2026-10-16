"""Banc de latence de la recherche web (reseau reel).

Pour chaque question, mesure le temps total de chercher_iteratif(), le nombre
d'appels a DuckDuckGo et la duree de chacun. Sert a decider si la seconde
passe est utile et combien elle coute.

Usage :  python scripts/banc_latence.py [--sortie resultats.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUESTIONS = [
    "prix de l or aujourd'hui",
    "qui est le président de la France",
    "résultats du match de football hier soir",
    "capitale de l'Australie",
    "cours du bitcoin maintenant",
    "dernière version de Python",
    "qui a inventé le téléphone",
    "ouverture de la bibliothèque municipale de Lyon",
    "taux d'inflation en France 2026",
    "film sorti au cinéma ce week-end",
    "quand est la prochaine éclipse solaire visible en Europe",
    "météo de demain à Bordeaux",
    "meilleur framework JavaScript en 2026",
    "histoire de la Tour Eiffel",
    "nouvelles de la NASA cette semaine",
    "recette de crêpes sans lait",
    "population de Tokyo",
    "date de sortie de la PS6",
    "coût d'un billet de train Paris Marseille",
    "définition de la photosynthèse",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sortie", default=None)
    args = parser.parse_args()

    from aura import memoire_web

    journal: list[dict] = []
    original = memoire_web.chercher

    def chronometre(requete, *a, **k):
        t0 = time.perf_counter()
        try:
            return original(requete, *a, **k)
        finally:
            journal.append({"requete": requete,
                            "duree": time.perf_counter() - t0})

    memoire_web.chercher = chronometre
    resultats = []
    try:
        for question in QUESTIONS:
            journal.clear()
            t0 = time.perf_counter()
            try:
                memoire_web.chercher_iteratif(question)
                erreur = None
            except Exception as e:          # noqa: BLE001 - on mesure, on continue
                erreur = str(e)
            total = time.perf_counter() - t0
            appels = [round(a["duree"], 3) for a in journal]
            resultats.append({"question": question, "total_s": round(total, 3),
                              "appels": len(journal), "durees_s": appels,
                              "erreur": erreur})
            print(f"{total:6.2f} s  {len(journal)} appel(s) {appels}  "
                  f"{question}")
    finally:
        memoire_web.chercher = original

    totaux = [r["total_s"] for r in resultats]
    doubles = sum(1 for r in resultats if r["appels"] >= 2)
    print(f"\nmediane {statistics.median(totaux):.2f} s, "
          f"max {max(totaux):.2f} s, questions a 2 appels : "
          f"{doubles}/{len(resultats)}")
    if args.sortie:
        Path(args.sortie).write_text(json.dumps(resultats, ensure_ascii=False,
                                                indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
