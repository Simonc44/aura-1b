"""Banc stable (52 questions) pour juger le controle de langue FR.

Chaque question a une liste de reponses acceptables, comparees par mot entier
(regex \\b...\\b) sur la sortie en minuscules. Conditions : sans vecteur,
rampe palier 10-14 a K=30, et trois controles aleatoires de meme energie.
Le but est de savoir si le banc discrimine, avant de juger une methode.

Usage : PYTHONPATH=. python scripts/banc_langue_cvec.py
"""
import re
import sys

import numpy as np
from llama_cpp import Llama

import rampe_couches_cvec as R
from aura.llama_cerveau import chemin_gguf
from test_cvec_prod import appliquer, generer, score_langue
from validation_rampe_cvec import FORME_PALIER, K, controle_aleatoire

# (question, reponses acceptables) ; aucune question des paires d'entrainement
BANC = [
    ("Q: What is the capital of Spain?\nA:", ["madrid"]),
    ("Q: What is the capital of Canada?\nA:", ["ottawa"]),
    ("Q: What is the capital of Egypt?\nA:", ["cairo"]),
    ("Q: What is the capital of Portugal?\nA:", ["lisbon"]),
    ("Q: What is the capital of Germany?\nA:", ["berlin"]),
    ("Q: What is the capital of Japan?\nA:", ["tokyo"]),
    ("Q: What is the capital of Italy?\nA:", ["rome"]),
    ("Q: What is the capital of Norway?\nA:", ["oslo"]),
    ("Q: What is the capital of Greece?\nA:", ["athens"]),
    ("Q: What is the capital of Poland?\nA:", ["warsaw"]),
    ("Q: What is the chemical symbol of iron?\nA:", ["fe"]),
    ("Q: What is the chemical symbol of gold?\nA:", ["au"]),
    ("Q: What is the chemical symbol of sodium?\nA:", ["na"]),
    ("Q: What is the chemical symbol of oxygen?\nA:", ["o"]),
    ("Q: What is the chemical symbol of silver?\nA:", ["ag"]),
    ("Q: Who composed the Ninth Symphony?\nA:", ["beethoven"]),
    ("Q: Who wrote the novel 1984?\nA:", ["orwell"]),
    ("Q: Who developed the theory of relativity?\nA:", ["einstein"]),
    ("Q: Who painted the Starry Night?\nA:", ["van gogh", "gogh"]),
    ("Q: Who wrote Pride and Prejudice?\nA:", ["austen"]),
    ("Q: Who wrote Romeo and Juliet?\nA:", ["shakespeare"]),
    ("Q: Who wrote The Odyssey?\nA:", ["homer"]),
    ("Q: Who discovered penicillin?\nA:", ["fleming"]),
    ("Q: Who wrote War and Peace?\nA:", ["tolstoy"]),
    ("Q: Who painted the Sistine Chapel ceiling?\nA:", ["michelangelo"]),
    ("Q: How many sides does a hexagon have?\nA:", ["6", "six"]),
    ("Q: How many continents are there?\nA:", ["7", "seven"]),
    ("Q: How many days are in a leap year?\nA:", ["366"]),
    ("Q: How many minutes are in an hour?\nA:", ["60", "sixty"]),
    ("Q: How many hours are in a day?\nA:", ["24", "twenty-four", "twenty four"]),
    ("Q: How many strings does a standard guitar have?\nA:", ["6", "six"]),
    ("Q: How many players are on a soccer team on the field?\nA:", ["11", "eleven"]),
    ("Q: Which planet is known as the Red Planet?\nA:", ["mars"]),
    ("Q: Which planet is closest to the Sun?\nA:", ["mercury"]),
    ("Q: What is the largest ocean on Earth?\nA:", ["pacific"]),
    ("Q: What color is the sky on a clear day?\nA:", ["blue"]),
    ("Q: What color are ripe bananas?\nA:", ["yellow"]),
    ("Q: What gas do plants absorb from the air?\nA:", ["carbon dioxide", "co2"]),
    ("Q: What is the hardest natural mineral?\nA:", ["diamond"]),
    ("Q: What is the freezing point of water in Celsius?\nA:", ["0", "zero"]),
    ("Q: What is the square root of 81?\nA:", ["9", "nine"]),
    ("Q: What is the currency of Japan?\nA:", ["yen"]),
    ("Q: What is the currency of the United Kingdom?\nA:", ["pound"]),
    ("Q: What language is spoken in Brazil?\nA:", ["portuguese"]),
    ("Q: What is the longest river in Africa?\nA:", ["nile"]),
    ("Q: What is the tallest mountain on Earth?\nA:", ["everest"]),
    ("Q: What organ pumps blood through the body?\nA:", ["heart"]),
    ("Q: What is the main gas in Earth's atmosphere?\nA:", ["nitrogen"]),
    ("Q: What is the smallest prime number?\nA:", ["2", "two"]),
    ("Q: What is the opposite of hot?\nA:", ["cold"]),
    ("Q: What do bees produce?\nA:", ["honey"]),
    ("Q: What animal is the largest on land?\nA:", ["elephant"]),
]
ECHELLE = 1.0


def juste(texte: str, reponses: list) -> bool:
    bas = texte.lower()
    return any(re.search(rf"\b{re.escape(r)}\b", bas) for r in reponses)


def main() -> int:
    print(f"cerveau : {chemin_gguf()} ; questions : {len(BANC)}", flush=True)
    R.K_NORME = K
    deltas = R.extraire_deltas()
    concept = R.vecteur_rampe(deltas, FORME_PALIER, ECHELLE)
    llm = Llama(model_path=chemin_gguf(), n_ctx=1024, verbose=False)

    def evaluer(nom: str, buf) -> None:
        if buf is not None:
            appliquer(llm, buf, R.N_EMBD, R.N_COUCHES)
        ok, fr, pads = 0, [], 0
        for question, reponses in BANC:
            texte = generer(llm, question, max_tokens=24)
            ok += juste(texte, reponses)
            fr.append(score_langue(texte))
            pads += "<pad>" in texte
        print(f"{nom:24s} justesse={ok}/{len(BANC)} "
              f"fr_moy={np.mean(fr):.2f} pad={pads}", flush=True)

    evaluer("aucun", None)
    evaluer("concept palier K30", concept)
    for seed in (0, 1, 2):
        evaluer(f"aleatoire s{seed}",
                controle_aleatoire(deltas, FORME_PALIER, seed))
    appliquer(llm, None, R.N_EMBD, R.N_COUCHES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
