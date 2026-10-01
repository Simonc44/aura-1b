"""Telecharge le detecteur de concepts dans modeles/minilm/.

all-MiniLM-L6-v2 quantifie INT8 (ONNX, ~23 Mo) : le micro-modele qui
transforme une phrase en vecteur 384-d en ~1-3 ms, 100 % local, zero LLM.

Trois composants s'en servent (voir aura/embeddings.py) :
- detecteur de concepts abstraits (aura/concepts.py) ;
- classificateur flou du routeur (aura/classificateur_flou.py) ;
- memoire semantique du cache de reponses (filtre_instantane).

Usage :
  python scripts/telecharger_minilm.py            # telecharge si absent
  python scripts/telecharger_minilm.py --force    # re-telecharge
"""
import argparse
import sys
from pathlib import Path

REPO = "Xenova/all-MiniLM-L6-v2"
# onnx/model_quantized.onnx = poids INT8 (modeles/ sert de depot local,
# comme les GGUF : jamais committes, voir .gitignore)
FICHIERS = {
    "model_quantized.onnx": "onnx/model_quantized.onnx",
    "tokenizer.json": "tokenizer.json",
}
DEST = Path(__file__).resolve().parent.parent / "modeles" / "minilm"


def main() -> int:
    ap = argparse.ArgumentParser(prog="telecharger_minilm")
    ap.add_argument("--force", action="store_true",
                    help="re-telecharge meme si les fichiers sont la")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    DEST.mkdir(parents=True, exist_ok=True)
    for local, distant in FICHIERS.items():
        cible = DEST / local
        if cible.exists() and not args.force:
            print(f"  deja la : {local} ({cible.stat().st_size / 1e6:.1f} Mo)")
            continue
        print(f"  telechargement : {REPO}/{distant} ...")
        chemin = hf_hub_download(REPO, distant,
                                 local_dir=str(DEST))
        source = Path(chemin)
        if source.resolve() != cible.resolve():
            # depot distant en sous-dossier (onnx/...) -> a plat ici
            source.replace(cible)
        print(f"  ok : {local} ({cible.stat().st_size / 1e6:.1f} Mo)")

    # verification rapide : la session ONNX s'ouvre vraiment
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(DEST / "model_quantized.onnx"),
                                    providers=["CPUExecutionProvider"])
        print(f"  ONNX pret : {len(sess.get_inputs())} entrees, "
              f"providers={sess.get_providers()}")
    except Exception as e:                # noqa: BLE001
        print(f"  [!] ONNX inutilisable : {e}")
        return 1
    print(f"[ok] detecteur de concepts pret dans {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
