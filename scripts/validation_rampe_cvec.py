"""Validation de la rampe par couche (K=30) sur des questions NEUVES.

K=30 a ete choisi sur les 10 questions de rampe_couches_cvec ; ici les
questions sont differentes et K est fixe a l'avance. Controle : pour chaque
couche, une direction aleatoire normalisee a la meme norme (K * w(L)) que le
concept, seed fixe.

Usage : PYTHONPATH=. python scripts/validation_rampe_cvec.py
"""
import sys

import numpy as np
from llama_cpp import Llama

import rampe_couches_cvec as R
from aura.llama_cerveau import chemin_gguf
from eval_cvec_prod import juste
from test_cvec_prod import appliquer, generer, score_langue

# (question, reponse attendue) : aucune n'est dans les paires d'entrainement
EVAL_NEUF = [
    ("Q: What is the capital of Spain?\nA:", "madrid"),
    ("Q: Who composed the Ninth Symphony?\nA:", "beethoven"),
    ("Q: What is the chemical symbol of iron?\nA:", "fe"),
    ("Q: What is the capital of Canada?\nA:", "ottawa"),
    ("Q: Who wrote the novel 1984?\nA:", "orwell"),
    ("Q: How many sides does a hexagon have?\nA:", "6"),
    ("Q: What is the capital of Egypt?\nA:", "cairo"),
    ("Q: Who developed the theory of relativity?\nA:", "einstein"),
    ("Q: What is the capital of Portugal?\nA:", "lisbon"),
    ("Q: Which planet is known as the Red Planet?\nA:", "mars"),
]
K = 30.0
FORME_PALIER = R.palier(10, 14)
FORME_TRIANGLE = R.triangle(8, 12, 16)


def controle_aleatoire(deltas: dict, forme, seed: int) -> np.ndarray:
    """Meme energie par couche que le concept, direction aleatoire."""
    rng = np.random.default_rng(seed)
    buf = np.zeros((R.N_COUCHES - 1) * R.N_EMBD, dtype=np.float32)
    for L in R.COUCHES:
        poids = forme(L)
        if poids == 0.0:
            continue
        d = rng.standard_normal(R.N_EMBD).astype(np.float32)
        d /= np.linalg.norm(d)
        buf[(L - 1) * R.N_EMBD:L * R.N_EMBD] = poids * K * d
    return buf


def main() -> int:
    print(f"cerveau : {chemin_gguf()}", flush=True)
    R.K_NORME = K
    deltas = R.extraire_deltas()
    llm = Llama(model_path=chemin_gguf(), n_ctx=1024, verbose=False)

    def evaluer(nom: str, buf) -> None:
        if buf is not None:
            appliquer(llm, buf, R.N_EMBD, R.N_COUCHES)
        ok, fr, pads = 0, [], 0
        for question, attendu in EVAL_NEUF:
            texte = generer(llm, question, max_tokens=24)
            ok += juste(texte, attendu)
            fr.append(score_langue(texte))
            pads += "<pad>" in texte
        print(f"{nom:28s} justesse={ok}/{len(EVAL_NEUF)} "
              f"fr_moy={np.mean(fr):.2f} pad={pads}", flush=True)

    evaluer("aucun", None)
    evaluer("K30 palier 10-14", R.vecteur_rampe(deltas, FORME_PALIER, 1.0))
    evaluer("K30 triangle 8-12-16", R.vecteur_rampe(deltas, FORME_TRIANGLE, 1.0))
    for seed in (0, 1):
        evaluer(f"alea triangle s{seed}",
                controle_aleatoire(deltas, FORME_TRIANGLE, seed))
    appliquer(llm, None, R.N_EMBD, R.N_COUCHES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
