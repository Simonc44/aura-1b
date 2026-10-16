"""Mesure le rappel, la justesse et les faux positifs du RAG binaire.

Le jeu d'evaluation est separe des exemples indexes. Les questions avec
`attendu: null` verifient que le RAG ne repond pas aux demandes hors corpus.
Ce script n'appelle ni le LLM ni le reseau.

Usage :
  .venv/Scripts/python.exe scripts/evaluer_rag.py
  .venv/Scripts/python.exe scripts/evaluer_rag.py --json resultats/rag.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import unicodedata
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
JEU = RACINE / "datasets" / "eval_rag_hors_echantillon.jsonl"


def normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", (texte or "").lower())
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return " ".join(texte.split())


def p95(valeurs: list[float]) -> float:
    if not valeurs:
        return 0.0
    valeurs = sorted(valeurs)
    return valeurs[round(.95 * (len(valeurs) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", metavar="CHEMIN", default="",
                        help="ecrit le rapport detaille")
    args = parser.parse_args()
    from aura import rag_binaire

    jeu = [json.loads(ligne)
           for ligne in JEU.read_text(encoding="utf-8").splitlines()
           if ligne.strip()]
    reponses, latences = [], []
    for item in jeu:
        t0 = time.perf_counter()
        reponse = rag_binaire.chercher(item["question"])
        latences.append((time.perf_counter() - t0) * 1000)
        attendu = item["attendu"]
        ok = ((reponse is None) if attendu is None else
              (reponse is not None and
               normaliser(attendu) in normaliser(reponse)))
        reponses.append({**item, "reponse": reponse, "ok": ok,
                         "latence_ms": round(latences[-1], 3)})

    positifs = [r for r in reponses if r["attendu"] is not None]
    negatifs = [r for r in reponses if r["attendu"] is None]
    exacts = sum(r["ok"] for r in positifs)
    rappeles = sum(r["reponse"] is not None for r in positifs)
    vrais_negatifs = sum(r["ok"] for r in negatifs)
    print(f"Index : {rag_binaire.statistiques()['documents']} documents")
    print(f"Questions de fait correctement servies : {exacts}/{len(positifs)}")
    print(f"Couverture des questions de fait : {rappeles}/{len(positifs)}")
    print(f"Abstentions correctes hors corpus : {vrais_negatifs}/{len(negatifs)}")
    print(f"Latence RAG p50/p95 : {statistics.median(latences):.2f}/"
          f"{p95(latences):.2f} ms")
    for r in reponses:
        if not r["ok"]:
            print(f"ECHEC [{r['categorie']}] {r['question']} -> "
                  f"{r['reponse']!r} (attendu {r['attendu']!r})")
    rapport = {
        "index_documents": rag_binaire.statistiques()["documents"],
        "faits_corrects": f"{exacts}/{len(positifs)}",
        "faits_couverts": f"{rappeles}/{len(positifs)}",
        "abstentions_correctes": f"{vrais_negatifs}/{len(negatifs)}",
        "latence_p50_ms": statistics.median(latences),
        "latence_p95_ms": p95(latences),
        "resultats": reponses,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(
            rapport, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Rapport : {args.json}")
    return 0 if all(r["ok"] for r in reponses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
