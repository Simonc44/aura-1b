"""Telecharge le GGUF Llama 3.2 1B Instruct Q4_K_M dans modeles/.

Usage : python scripts/telecharger_llama.py
"""
import sys
from pathlib import Path

REPO = "bartowski/Llama-3.2-1B-Instruct-GGUF"
FICHIER = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
DEST = Path(__file__).resolve().parent.parent / "modeles"


def main() -> int:
    from huggingface_hub import hf_hub_download
    DEST.mkdir(exist_ok=True)
    print(f"Telechargement de {REPO}/{FICHIER} (~807 Mo)...")
    chemin = hf_hub_download(REPO, FICHIER, local_dir=str(DEST))
    print(f"OK : {chemin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
