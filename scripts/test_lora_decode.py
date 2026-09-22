"""Test decode LoRA minimal : charge base+adaptateur, genere quelques tokens.

Usage : python test_lora_decode.py MODELE.GGUF ADAPTATEUR.GGUF
Succes -> exit 0 + DECODE_OK ; echec -> exit != 0.
"""
import sys

from llama_cpp import Llama


def main() -> int:
    modele, adaptateur = sys.argv[1], sys.argv[2]
    try:
        llm = Llama(
            model_path=modele,
            n_ctx=256,
            n_threads=4,
            verbose=False,
            lora_path=adaptateur,
        )
        out = llm("Dis bonjour en un mot.", max_tokens=4)
    except Exception as exc:  # noqa: BLE001
        print(f"ECHEC: {exc}")
        return 1
    texte = out["choices"][0]["text"].strip()[:60]
    print(f"DECODE_OK: {texte!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
