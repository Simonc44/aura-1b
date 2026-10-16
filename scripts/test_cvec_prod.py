"""Test du control vector langue_fr.cvec sur le cerveau GGUF de production.

Charge le meme GGUF qu'Aura (aura.llama_cerveau.chemin_gguf), applique le
vecteur de la couche 16 via llama_set_adapter_cvec, puis compare :
  - sans intervention ;
  - vecteur concept (adaptateurs/langue_fr.cvec) ;
  - vecteur aleatoire de meme norme (controle, seed fixe).
Prompts tenus a l'ecart des paires d'entrainement.

Usage : PYTHONPATH=. python scripts/test_cvec_prod.py
"""
import ctypes
import os
import sys

import numpy as np
import gguf
import llama_cpp
from llama_cpp import Llama

from aura.llama_cerveau import chemin_gguf

CVEC = os.environ.get("AURA_CVEC_TEST",
                      os.path.join("adaptateurs", "langue_fr_heretic.cvec"))
COUCHE = 16
PROMPTS = [
    "Q: What is the capital of Germany?\nA:",
    "Q: Who painted the Mona Lisa?\nA:",
]
MOTS_FR = {"le", "la", "les", "de", "des", "est", "une", "un", "et", "qui",
           "que", "du", "pas", "c'est", "réponse", "reponse"}
MOTS_EN = {"the", "is", "of", "and", "a", "an", "what", "who", "how", "many",
           "that", "it", "not"}


def score_langue(texte: str) -> float:
    mots = [m.strip(".,:?!\"'").lower() for m in texte.split()]
    fr = sum(m in MOTS_FR for m in mots)
    en = sum(m in MOTS_EN for m in mots)
    return fr / (fr + en) if fr + en else 0.5


def lire_vecteurs(chemin: str) -> tuple[np.ndarray, int]:
    lecteur = gguf.GGUFReader(chemin)
    tenseurs = {t.name: np.array(t.data, dtype=np.float32) for t in lecteur.tensors}
    n_couches = max(int(n.split(".")[1]) for n in tenseurs) + 1
    buf = np.concatenate([tenseurs[f"direction.{il}"] for il in range(1, n_couches)])
    return buf.astype(np.float32), tenseurs[f"direction.{COUCHE}"].shape[0]


def appliquer(llm, buf: np.ndarray | None, n_embd: int, n_couches: int) -> int:
    ctx = llm._ctx.ctx
    if buf is None:
        return llama_cpp.llama_set_adapter_cvec(ctx, None, 0, 0, 0, 0)
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return llama_cpp.llama_set_adapter_cvec(
        ctx, ptr, len(buf), n_embd, 1, n_couches - 1)


def generer(llm, prompt: str, max_tokens: int = 40) -> str:
    llm.reset()
    sortie = llm(prompt, max_tokens=max_tokens, temperature=0.0)
    return sortie["choices"][0]["text"]


def main() -> int:
    chemin = chemin_gguf()
    print(f"cerveau : {chemin}")
    buf, n_embd = lire_vecteurs(CVEC)
    n_couches = len(buf) // n_embd + 1
    concept = buf.copy()
    concept_seg = concept[(COUCHE - 1) * n_embd: COUCHE * n_embd]
    print(f"n_embd={n_embd} couches={n_couches} norme couche {COUCHE}="
          f"{np.linalg.norm(concept_seg):.2f}")

    rng = np.random.default_rng(0)
    alea = np.zeros_like(buf)
    bruit = rng.standard_normal(n_embd).astype(np.float32)
    bruit *= np.linalg.norm(concept_seg) / np.linalg.norm(bruit)
    alea[(COUCHE - 1) * n_embd: COUCHE * n_embd] = bruit

    llm = Llama(model_path=chemin, n_ctx=1024, verbose=False)
    cas = [("aucun", None), ("concept x0.5", concept * 0.5),
           ("concept x1", concept), ("aleatoire", alea),
           ("concept x2", concept * 2)]
    for nom, vec in cas:
        r = appliquer(llm, vec, n_embd, n_couches) if vec is not None \
            else appliquer(llm, None, n_embd, n_couches)
        print(f"[{nom}] set_adapter_cvec -> {r}")
        for prompt in PROMPTS:
            texte = generer(llm, prompt)
            print(f"  fr={score_langue(texte):.2f} pad={'<pad>' in texte} {texte!r}")
    appliquer(llm, None, n_embd, n_couches)
    return 0


if __name__ == "__main__":
    sys.exit(main())
