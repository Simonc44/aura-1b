"""Sonde d'interpretabilite : « logit lens » sur un grand modele HF.

Pour chaque couche, on projette l'etat cache du dernier token sur le
vocabulaire (norme finale + lm_head) et on relève le token le plus probable.
On voit ainsi a quel etage la reponse se forme (ex. chiffre juste avant la
derniere couche). Sert a decider quelles capacites Aura doit combler.

Usage :
    python scripts/sonde_interp.py                      # prompts par defaut
    python scripts/sonde_interp.py --modele unsloth/gemma-3-4b-it --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys

DEFAUT_MODELE = "unsloth/gemma-3-4b-it"

# Un prompt par capacite a sonder (arithmetique, fait, identite, refus).
PROMPTS = [
    "Q: Combien font 17 * 23 ?\nR:",
    "Q: Combien font 17 * 23 ?\nR: 17 * 23 =",
    "Q: Quelle est la capitale de l'Australie ?\nR:",
    "Q: Qui es-tu ?\nR:",
    "Q: Est-ce que la Terre tourne autour du Soleil ?\nR:",
]


def charger(modele: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(modele)
    mod = AutoModelForCausalLM.from_pretrained(
        modele, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    mod.eval()
    return tok, mod


def _norme_finale(mod):
    """Norme finale du decodeur (chemin differe selon Gemma-3 texte ou multimodal)."""
    for chemin in ("model.norm", "model.language_model.norm"):
        cible = mod
        try:
            for attr in chemin.split("."):
                cible = getattr(cible, attr)
            return cible
        except AttributeError:
            continue
    raise AttributeError("norme finale introuvable (architecture inconnue)")


def logit_lens(tok, mod, prompt: str, top: int = 3) -> list[dict]:
    """Par couche : top tokens du dernier emplacement, projetes sur le vocabulaire."""
    import torch

    entree = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        sortie = mod(**entree, output_hidden_states=True)
    norme = _norme_finale(mod)
    lm_head = mod.lm_head
    resultats = []
    for i, h in enumerate(sortie.hidden_states):
        dernier = h[0, -1]
        with torch.no_grad():
            logits = lm_head(norme(dernier))
            proba = torch.softmax(logits.float(), dim=-1)
            vals, idx = torch.topk(proba, top)
        resultats.append({
            "couche": i,
            "top": [{"token": tok.decode([int(j)]), "p": round(float(v), 4)}
                    for v, j in zip(vals, idx)],
        })
    return resultats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--modele", default=DEFAUT_MODELE)
    p.add_argument("--json", help="ecrit les resultats dans ce fichier")
    args = p.parse_args(argv)

    tok, mod = charger(args.modele)
    tout = {}
    for prompt in PROMPTS:
        couches = logit_lens(tok, mod, prompt)
        tout[prompt] = couches
        derniere = couches[-1]["top"][0]
        print(f"{prompt!r} -> sortie finale : {derniere['token']!r} "
              f"(p={derniere['p']})")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(tout, f, ensure_ascii=False, indent=2)
        print(f"ecrit : {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
