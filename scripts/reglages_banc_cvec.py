"""Reglages stables sur le banc de langue, avec separation dev / test.

Le banc (banc_langue_cvec.BANC) est coupe en deux : les questions d'indice
pair servent au reglage (dev), celles d'indice impair a la mesure finale
(test). Le meilleur reglage est choisi sur dev uniquement, puis mesure sur
test avec les controles (aucun vecteur, aleatoire de meme energie).

Reglages balayes :
  - K : norme commune des deltas (rampe_couches_cvec.K_NORME) ;
  - repeat_penalty : penalite de repetition llama.cpp (1.0 = desactivee).
La generation reste deterministe (temperature=0), comme dans le banc.

Usage : PYTHONPATH=. python scripts/reglages_banc_cvec.py
"""
import sys
from dataclasses import dataclass

import numpy as np
from llama_cpp import Llama

import rampe_couches_cvec as R
from aura.llama_cerveau import chemin_gguf
from banc_langue_cvec import BANC, juste
from test_cvec_prod import appliquer, score_langue
from validation_rampe_cvec import FORME_PALIER, controle_aleatoire

K_VALEURS = (10, 20, 30)
PENALITES = (1.0, 1.1)


@dataclass(frozen=True)
class Reglage:
    k: float
    repeat_penalty: float

    def nom(self) -> str:
        return f"K={self.k:g} rep={self.repeat_penalty:g}"


def generer_reglee(llm, prompt: str, repeat_penalty: float,
                   max_tokens: int = 24) -> str:
    llm.reset()
    sortie = llm(prompt, max_tokens=max_tokens, temperature=0.0,
                 repeat_penalty=repeat_penalty)
    return sortie["choices"][0]["text"]


def evaluer(llm, questions, buf, reglage: Reglage) -> tuple[int, int, float]:
    if buf is not None:
        appliquer(llm, buf, R.N_EMBD, R.N_COUCHES)
    else:
        appliquer(llm, None, R.N_EMBD, R.N_COUCHES)
    ok, fr, pads = 0, [], 0
    for question, reponses in questions:
        texte = generer_reglee(llm, question, reglage.repeat_penalty)
        ok += juste(texte, reponses)
        fr.append(score_langue(texte))
        pads += "<pad>" in texte
    return ok, pads, float(np.mean(fr))


def main() -> int:
    dev = BANC[0::2]
    test = BANC[1::2]
    print(f"cerveau : {chemin_gguf()} ; dev={len(dev)} test={len(test)}",
          flush=True)
    llm = Llama(model_path=chemin_gguf(), n_ctx=1024, verbose=False)

    # 1) Balayage sur dev : K x penalite. Le vecteur depend de K, pas de la
    #    penalite, donc on calcule chaque vecteur une seule fois.
    resultats = []
    for k in K_VALEURS:
        R.K_NORME = k
        deltas = R.extraire_deltas()
        concept = R.vecteur_rampe(deltas, FORME_PALIER, 1.0)
        for pen in PENALITES:
            reglage = Reglage(k, pen)
            ok, pads, fr = evaluer(llm, dev, concept, reglage)
            print(f"dev {reglage.nom():18s} justesse={ok}/{len(dev)} "
                  f"fr={fr:.2f} pad={pads}", flush=True)
            resultats.append((ok, fr, reglage, pads, concept))

    # Choix sur dev : justesse d'abord, puis langue, sans <pad>.
    valides = [r for r in resultats if r[3] == 0]
    if not valides:
        print("ECHEC : aucun reglage sans <pad> sur dev", flush=True)
        return 1
    _, _, choisi, _, concept_choisi = max(valides, key=lambda r: (r[0], r[1]))
    print(f"reglage retenu sur dev : {choisi.nom()}", flush=True)

    # 2) Mesure finale sur test, avec controles.
    R.K_NORME = choisi.k
    deltas = R.extraire_deltas()
    concept = R.vecteur_rampe(deltas, FORME_PALIER, 1.0)
    alea = controle_aleatoire(deltas, FORME_PALIER, 0)
    conditions = [
        ("aucun", None),
        (f"concept {choisi.nom()}", concept),
        ("aleatoire s0 (controle)", alea),
    ]
    for nom, buf in conditions:
        ok, pads, fr = evaluer(llm, test, buf, choisi)
        print(f"test {nom:30s} justesse={ok}/{len(test)} "
              f"fr={fr:.2f} pad={pads}", flush=True)
    appliquer(llm, None, R.N_EMBD, R.N_COUCHES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
