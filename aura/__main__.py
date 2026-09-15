"""CLI : python -m aura "question" [--x ... --y ...] [--demo]

Sans LLM installe, le pipeline reste utilisable (web + formule exacte).
"""
import argparse
import json
import sys

from .orchestrateur import Aura1B


def _demo(ia: Aura1B) -> int:
    """Demo : decouvrir Y = X^3 a partir des donnees, plus une question web."""
    X = [[0.0], [0.25], [0.5], [0.75], [1.0]]
    y = [0.0, 0.015625, 0.125, 0.421875, 1.0]  # Y = X^3 sur [0,1]
    r = ia.executer_detaille("Donne-moi les dernieres decouvertes en physique et resous la puissance de ces chiffres.", X, y)
    print("=" * 60)
    print(" ✨ AURA-1B — rapport neuro-symbolique")
    print("=" * 60)
    print(f"Web sollicite       : {r['web_actif']}")
    if r["contexte_web"]:
        print(f"Contexte web        :\n{r['contexte_web']}\n")
    print(f"Formule exacte PGS  : Y = {r['formule']}  (erreur {r['erreur_pgs']:.3g})")
    print(f"Reponse finale      :\n{r['reponse']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="aura")
    p.add_argument("question", nargs="?", default="", help="question en langage naturel")
    p.add_argument("--x", help="liste JSON de vecteurs, ex. '[[1],[2],[3]]'")
    p.add_argument("--y", help="liste JSON de valeurs, ex. '[1,8,27]'")
    p.add_argument("--hote", default="http://localhost:11434")
    p.add_argument("--modele", default="qwen2.5:1.5b-instruct", help="modele local (Ollama)")
    p.add_argument("--cerveau", choices=["ollama", "mamba"], default="ollama",
                   help="mamba = SSM lineaire (repli auto sur ollama)")
    p.add_argument("--grande", action="store_true",
                   help="mamba 1.4B au lieu de 370M (necessite 16 Go RAM)")
    p.add_argument("--demo", action="store_true", help="lancer la demo integree")
    args = p.parse_args(argv)

    ia = Aura1B(hote=args.hote, modele=args.modele, cerveau=args.cerveau,
                mamba_grande=args.grande)
    if args.demo or (not args.question and not (args.x and args.y)):
        return _demo(ia)

    X = json.loads(args.x) if args.x else None
    y = json.loads(args.y) if args.y else None
    print(ia.executer(args.question, X, y))
    return 0


if __name__ == "__main__":
    sys.exit(main())
