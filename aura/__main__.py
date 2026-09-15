"""CLI Aura-1B : systeme neuro-symbolique autonome (sans Ollama).

Usage :
  python -m aura "question"
  python -m aura --cerveau mamba "question"
  python -m aura --cerveau rwkv "question"
  python -m aura --demo
"""
import argparse
import json
import sys

from .orchestrateur import Aura1B


def _demo(ia: Aura1B) -> int:
    X = [[0.0], [0.25], [0.5], [0.75], [1.0]]
    y = [0.0, 0.015625, 0.125, 0.421875, 1.0]
    r = ia.executer_detaille(
        "Dernieres decouvertes en physique et resous X cube", X, y)
    print("=" * 60)
    print(" AURA-1B — MoE neuro-symbolique autonome")
    print("=" * 60)
    print(f"Experts actives : {r['experts']}")
    print(f"Cerveau choisi  : {r['cerveau_choisi']}")
    if r["contexte_web"]:
        print(f"\nWeb :\n{r['contexte_web'][:300]}")
    print(f"\nFormule PGS    : Y = {r['formule']}  (erreur {r['erreur_pgs']:.3g})")
    print(f"\nReponse :\n{r['reponse'][:300]}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="aura")
    p.add_argument("question", nargs="?", default="")
    p.add_argument("--x", help="liste JSON de vecteurs, ex. '[[1],[2],[3]]'")
    p.add_argument("--y", help="liste JSON de valeurs, ex. '[1,8,27]'")
    p.add_argument("--cerveau", choices=["mamba", "rwkv"], default="rwkv",
                   help="rwkv = langage (defaut), mamba = logique/code")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args(argv)

    ia = Aura1B()
    if args.demo or (not args.question and not (args.x and args.y)):
        return _demo(ia)

    X = json.loads(args.x) if args.x else None
    y = json.loads(args.y) if args.y else None
    print(ia.executer(args.question, X, y))
    return 0


if __name__ == "__main__":
    sys.exit(main())
