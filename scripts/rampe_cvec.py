"""Rampe d'echelle par couche du control vector FR (production).

Au lieu d'une echelle unique, chaque couche de la bande recoit un poids
w_L (rampe lineaire ou palier) applique au vecteur de concept de cette couche :
    contribution_L = echelle * w_L * concept_L
Les profils sont compares sur les memes 10 questions tenues a l'ecart que
eval_cvec_prod.py (justesse, langue, <pad>).

Usage : PYTHONPATH=. python scripts/rampe_cvec.py
"""
import sys

import numpy as np
from llama_cpp import Llama

from aura.llama_cerveau import chemin_gguf
from balayage_couches_cvec import COUCHES, N_COUCHES, N_EMBD, extraire_concepts
from eval_cvec_prod import EVAL, juste
from test_cvec_prod import appliquer, generer, score_langue


def buffer_profil(concepts: dict, poids: dict) -> np.ndarray:
    """Buffer dont chaque couche c porte poids[c] * concept_c (autres a zero)."""
    buf = np.zeros((N_COUCHES - 1) * N_EMBD, dtype=np.float32)
    for c, w in poids.items():
        buf[(c - 1) * N_EMBD:c * N_EMBD] = concepts[c] * w
    return buf


def rampe(bande, echelle: float) -> dict:
    """Poids lineaires de 0 (premiere couche) a echelle (derniere couche)."""
    n = len(bande) - 1
    return {c: echelle * i / n for i, c in enumerate(bande)}


def palier(bande, echelle: float) -> dict:
    return {c: echelle for c in bande}


PROFILS = [
    ("plat c10 x1", {10: 1.0}),
    ("plat c12 x1", {12: 1.0}),
    ("palier 10-14 x0.5", palier(COUCHES, 0.5)),
    ("rampe 10-14 x1", rampe(COUCHES, 1.0)),
    ("rampe 10-14 x0.5", rampe(COUCHES, 0.5)),
]


def main() -> int:
    print(f"cerveau : {chemin_gguf()}")
    concepts = extraire_concepts()
    llm = Llama(model_path=chemin_gguf(), n_ctx=1024, verbose=False)

    appliquer(llm, None, N_EMBD, N_COUCHES)
    ok, fr, pads = 0, [], 0
    for question, attendu in EVAL:
        texte = generer(llm, question, max_tokens=24)
        ok += juste(texte, attendu)
        fr.append(score_langue(texte))
        pads += "<pad>" in texte
    print(f"{'aucun':20s} justesse={ok}/{len(EVAL)} "
          f"fr_moy={np.mean(fr):.2f} pad={pads}", flush=True)

    for nom, poids in PROFILS:
        appliquer(llm, buffer_profil(concepts, poids), N_EMBD, N_COUCHES)
        ok, fr, pads = 0, [], 0
        for question, attendu in EVAL:
            texte = generer(llm, question, max_tokens=24)
            ok += juste(texte, attendu)
            fr.append(score_langue(texte))
            pads += "<pad>" in texte
        print(f"{nom:20s} justesse={ok}/{len(EVAL)} "
              f"fr_moy={np.mean(fr):.2f} pad={pads}", flush=True)
    appliquer(llm, None, N_EMBD, N_COUCHES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
