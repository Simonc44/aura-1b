"""Affiche l'architecture REELLE d'un GGUF — ce que le fichier declare.

La lecture vit dans `aura/gguf_meta.py` (l'adaptateur LoRA a besoin du meme
verdict que l'humain qui lit ce diagnostic, donc une seule implementation).
Ce script n'est que la sortie lisible.

Usage :
    python scripts/lire_gguf.py modeles/modele.gguf [--json] [--tout]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura import gguf_meta  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lire_gguf")
    ap.add_argument("chemin")
    ap.add_argument("--json", action="store_true",
                    help="sortie machine (une ligne JSON)")
    ap.add_argument("--tout", action="store_true",
                    help="toutes les metadonnees, pas seulement l'architecture")
    args = ap.parse_args(argv)

    if args.json:
        print(json.dumps(gguf_meta.architecture(args.chemin),
                         ensure_ascii=False))
        return 0

    arc = gguf_meta.architecture(args.chemin)
    print(f"--- {Path(args.chemin).name}")
    if not arc:
        print("    (illisible : fichier absent ou tronque)")
        return 1
    for cle, val in arc.items():
        print(f"    {cle:24s} = {val}")
    if args.tout:
        print("--- toutes les metadonnees")
        for cle, val in gguf_meta.lire(args.chemin, tout=True).items():
            print(f"    {cle:48s} = {str(val)[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
