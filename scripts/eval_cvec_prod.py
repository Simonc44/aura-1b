"""Evaluation elargie du control vector langue_fr_heretic.cvec (production).

Dix questions tenues a l'ecart des paires d'entrainement. Pour chaque
condition (sans vecteur, concept x echelle, aleatoire de meme norme) on
mesure :
  - justesse : la bonne reponse (sous-chaine attendue) apparait-elle ;
  - langue   : part de mots fonctionnels francais (score grossier) ;
  - pad      : presence de <pad> (doit rester 0).

Usage : PYTHONPATH=. python scripts/eval_cvec_prod.py
"""
import sys

import numpy as np

from aura.llama_cerveau import chemin_gguf
from llama_cpp import Llama
from test_cvec_prod import (CVEC, COUCHE, appliquer, generer, lire_vecteurs,
                            score_langue)

# (question, sous-chaine attendue dans la reponse)
EVAL = [
    ("Q: What is the capital of Germany?\nA:", "berlin"),
    ("Q: Who painted the Mona Lisa?\nA:", "vinci"),
    ("Q: What is the capital of Japan?\nA:", "tokyo"),
    ("Q: What is the chemical symbol of gold?\nA:", "au"),
    ("Q: Who wrote Romeo and Juliet?\nA:", "shakespeare"),
    ("Q: What is the capital of Italy?\nA:", "rome"),
    ("Q: What is the largest ocean on Earth?\nA:", "pacific"),
    ("Q: What color is the sky on a clear day?\nA:", "blue"),
    ("Q: Who discovered penicillin?\nA:", "fleming"),
    ("Q: How many continents are there?\nA:", "7"),
]
ECHELLES = (0.5, 0.75, 1.0, 1.25, 1.5)
MAX_TOKENS = 24


def juste(texte: str, attendu: str) -> bool:
    return attendu in texte.lower()


def main() -> int:
    buf, n_embd = lire_vecteurs(CVEC)
    n_couches = len(buf) // n_embd + 1
    seg = slice((COUCHE - 1) * n_embd, COUCHE * n_embd)
    concept = np.zeros_like(buf)
    concept[seg] = buf[seg]
    norme = float(np.linalg.norm(concept[seg]))
    rng = np.random.default_rng(0)
    bruit = rng.standard_normal(n_embd).astype(np.float32)
    bruit *= norme / np.linalg.norm(bruit)
    alea = np.zeros_like(buf)
    alea[seg] = bruit

    print(f"cerveau : {chemin_gguf()}")
    print(f"norme couche {COUCHE} = {norme:.2f}, prompts = {len(EVAL)}")
    llm = Llama(model_path=chemin_gguf(), n_ctx=1024, verbose=False)

    conditions = [("aucun", None)]
    conditions += [(f"concept x{e}", concept * e) for e in ECHELLES]
    conditions += [("aleatoire x1", alea), ("aleatoire x1.5", alea * 1.5)]

    for nom, vec in conditions:
        appliquer(llm, vec, n_embd, n_couches)
        ok, fr, pads = 0, [], 0
        for question, attendu in EVAL:
            texte = generer(llm, question, max_tokens=MAX_TOKENS)
            ok += juste(texte, attendu)
            fr.append(score_langue(texte))
            pads += "<pad>" in texte
        print(f"{nom:16s} justesse={ok}/{len(EVAL)} "
              f"fr_moy={np.mean(fr):.2f} pad={pads}")
    appliquer(llm, None, n_embd, n_couches)
    return 0


if __name__ == "__main__":
    sys.exit(main())
