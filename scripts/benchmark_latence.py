"""Benchmark de latence p50/p95 par chemin du pipeline.

Le README cite des chiffres ponctuels ; ce script mesure sur un lot de
questions les QUANTILES par chemin (niveau 0 / cache, RAG binaire,
identite, memoire). Objectif : prouver objectivement que le Dynamic
Compute ne degrade pas les questions simples.

Par defaut SANS LLM : reproductible, utilisable avant chaque publication.
--llm ajoute les chemins couteux (web, redaction) — lent, a lancer a part.

Le cache semantique persistant (aura/.cache_reponses.jsonl) est sauvegarde
au demarrage et restaure a la fin : le benchmark ne le pollue pas.

Usage :
  uv run python scripts/benchmark_latence.py             # sans LLM, ~1 min
  uv run python scripts/benchmark_latence.py --llm       # + web et redaction
  uv run python scripts/benchmark_latence.py --n 30      # n repetitions/chemin
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))


def _p50_p95(valeurs: list[float]) -> tuple[float, float]:
    s = sorted(valeurs)
    return statistics.median(s), s[round(0.95 * (len(s) - 1))]


def _mesurer(libelle: str, fn, n: int) -> None:
    """Mesure n fois fn(i) et imprime p50/p95 en ms."""
    latences: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        try:
            fn(i)
        except Exception as e:
            print(f"  {libelle:<18} ERREUR: {e}")
            return
        latences.append((time.perf_counter() - t0) * 1000)
    p50, p95 = _p50_p95(latences)
    print(f"  {libelle:<18} p50 {p50:>8.1f} ms   p95 {p95:>8.1f} ms   "
          f"(n={len(latences)})")


def main() -> int:
    parseur = argparse.ArgumentParser(
        description="Benchmark de latence p50/p95 par chemin du pipeline")
    parseur.add_argument("--n", type=int, default=50,
                         help="repetitions par chemin (defaut 50)")
    parseur.add_argument("--llm", action="store_true",
                         help="inclut les chemins couteux (web, redaction)")
    args = parseur.parse_args()

    from aura import filtre_instantane, rag_binaire
    from aura.memoire_conversation import MemoireConversation
    from aura.orchestrateur import Aura1B

    # le cache semantique persistant n'est pas pollue par le benchmark :
    # sauvegarde au demarrage, restaure a la fin (meme en cas d'erreur)
    cache = Path(filtre_instantane.__file__).parent / ".cache_reponses.jsonl"
    sauvegarde = None
    if cache.exists():
        sauvegarde = cache.with_suffix(".jsonl.bench-bak")
        shutil.copy2(cache, sauvegarde)
    try:
        print(f"[benchmark] n={args.n} par chemin "
              f"{'(avec LLM/web)' if args.llm else '(sans LLM)'}\n")

        print("Chemins sans LLM :")
        # maths directes : question UNIQUE a chaque iteration (jamais cache)
        _mesurer("niveau0-math", lambda i: filtre_instantane.repondre(
            f"calcule {13 + i} plus {27 + i}"), args.n)
        # cache semantique : meme question a chaque iteration (hit des la 2e)
        q_cache = "combien font 7 fois 8"
        filtre_instantane.repondre(q_cache)          # amorce le cache
        _mesurer("niveau0-cache", lambda i: filtre_instantane.repondre(q_cache),
                 args.n)
        _mesurer("rag-binaire", lambda i: rag_binaire.chercher(
            "qu'est-ce que la photosynthese"), args.n)
        # identite : pipeline complet, reponse constante (0 ms de LLM).
        # L'orchestrateur est construit HORS mesure (le 1er appel entraine
        # le routeur ~4 s : cout d'init, pas de reponse).
        print("  (construction de l'orchestrateur...)")
        aura = Aura1B()
        _mesurer("identite-pipeline", lambda i: aura.executer("qui t'a cree"),
                 max(args.n // 5, 1))

        # memoire : instance isolee (jamais la vraie .conversation.jsonl)
        with tempfile.TemporaryDirectory() as tmp:
            memoire = MemoireConversation(chemin=Path(tmp) / "bench.jsonl")
            compteur = 0

            def _tour_memoire(_i: int) -> None:
                nonlocal compteur
                compteur += 1
                memoire.ajouter("user", f"je m'appelle Pierre numero {compteur}")
                memoire.etat_json()
                memoire.rechercher("Pierre")

            _mesurer("memoire-1-tour", _tour_memoire, args.n)

        if args.llm:
            from aura import memoire_web
            from aura.llama_cerveau import generer as llama_generer
            print("\nChemins avec LLM (lent, peu de repetitions) :")
            _mesurer("web-factuel", lambda i: memoire_web.chercher(
                "quelle est la capitale de l'Australie",
                max_resultats=2, timeout=6), min(args.n // 10, 5) or 1)
            _mesurer("riche-redaction", lambda i: llama_generer(
                "Explique pourquoi le ciel est bleu", max_tokens=200), 2)
    finally:
        if sauvegarde is not None:
            shutil.move(sauvegarde, cache)

    print("\nReproduire : uv run python scripts/benchmark_latence.py"
          + (" --llm" if args.llm else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
