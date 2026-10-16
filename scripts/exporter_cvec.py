"""Exporte le vecteur de concept FR au format control vector GGUF (llama.cpp).

Le vecteur est calcule dans le courant residuel de gemma-3-1b-it (TransformerLens),
couche 16 (blocks.16.hook_resid_post). Il est ecrit dans adaptateurs/langue_fr.cvec
sous la forme attendue par llama.cpp :
  - general.architecture = "controlvector"
  - controlvector.model_hint, controlvector.layer_count
  - un tenseur F32 de taille n_embd par couche, nomme "direction.<il>"
    (il = 1 .. n_layers-1 ; seules les couches hors cible sont a zero).

Puis le fichier est relu et compare, tenseur par tenseur, au vecteur source.

Usage : PYTHONPATH=. python scripts/exporter_cvec.py
"""
import os
import sys

import numpy as np
import torch
import gguf

from scripts.intervention_tl import charger, vecteur_concept

COUCHE = 16
SORTIE = os.path.join("adaptateurs", "langue_fr.cvec")


def ecrire(vec: np.ndarray, n_couches: int, chemin: str) -> None:
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    ecrivain = gguf.GGUFWriter(chemin, arch="controlvector")
    ecrivain.add_string("controlvector.model_hint", "gemma3")
    ecrivain.add_int32("controlvector.layer_count", n_couches)
    n_embd = vec.shape[0]
    for il in range(1, n_couches):
        if il == COUCHE:
            dir_il = vec.astype(np.float32)
        else:
            dir_il = np.zeros(n_embd, dtype=np.float32)
        ecrivain.add_tensor(f"direction.{il}", dir_il)
    ecrivain.write_header_to_file()
    ecrivain.write_kv_data_to_file()
    ecrivain.write_tensors_to_file()
    ecrivain.close()


def relire(chemin: str) -> dict[str, np.ndarray]:
    lecteur = gguf.GGUFReader(chemin)
    return {t.name: np.array(t.data, dtype=np.float32) for t in lecteur.tensors}


def main() -> int:
    modele, tok = charger()
    vec = vecteur_concept(modele, tok, COUCHE).detach().cpu().float().numpy()
    n_couches = modele.cfg.n_layers
    print(f"vecteur couche {COUCHE} : n_embd={vec.shape[0]} norme={np.linalg.norm(vec):.4f}")

    ecrire(vec, n_couches, SORTIE)
    print(f"ecrit : {SORTIE} ({os.path.getsize(SORTIE)} octets)")

    relu = relire(SORTIE)
    cible = relu[f"direction.{COUCHE}"]
    if not np.array_equal(cible, vec.astype(np.float32)):
        print("ECHEC : direction relue differente de la source")
        return 1
    attendus = {f"direction.{il}" for il in range(1, n_couches)}
    if set(relu) != attendus:
        print("ECHEC : noms de tenseurs inattendus")
        return 1
    if any(np.any(relu[f"direction.{il}"] != 0)
           for il in range(1, n_couches) if il != COUCHE):
        print("ECHEC : une couche hors cible n'est pas nulle")
        return 1
    print("verification OK : direction relue identique a la source, autres couches nulles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
