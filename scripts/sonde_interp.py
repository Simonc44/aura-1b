"""Sonde d'interpretabilite : « logit lens » sur un grand modele HF.

Pour chaque couche, on projette l'etat cache du dernier token sur le
vocabulaire (norme finale + lm_head) et on relève le token le plus probable.
On voit ainsi a quel etage la reponse se forme (ex. chiffre juste avant la
derniere couche). Sert a decider quelles capacites Aura doit combler.

Usage :
    python scripts/sonde_interp.py                      # prompts par defaut
    python scripts/sonde_interp.py --modele unsloth/gemma-3-4b-it --json out.json

A SAVOIR AVANT DE LIRE UN RESULTAT :

* `torch` et `transformers` ne sont PAS des dependances du projet (`uv.lock`),
  donc la sonde ne tourne que dans un environnement qui les a, par exemple :
  `uv run --with torch --with transformers python scripts/sonde_interp.py`.
* Le GABARIT DE CONVERSATION n'est pas applique : le prompt part brut. Le
  modele est donc sonde hors de son format d'entrainement (mesure du 16 oct.
  2026 : sa derniere couche predit l'amorce d'une PHRASE au lieu de la reponse
  pourtant formee deux couches plus bas). A relire avant toute conclusion sur
  la formulation.
* Les premieres couches ne veulent RIEN dire : le flux residuel n'y est pas
  encore aligne sur le vocabulaire de sortie (mesure : des fragments de tamil,
  bengali, cyrillique autour de p=0,5). Seules les ~10 dernieres sont lisibles.
* Le JSON ecrit porte desormais sa provenance (`_methode`) : modele, date,
  dtype, top, nombre de couches. Un resultat qu'on ne peut ni dater ni
  rattacher a un modele n'est pas une mesure — cf. `resultats/NOTE_sonde_gemma4b.md`.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

DEFAUT_MODELE = "unsloth/gemma-3-4b-it"

#: Nombre de tokens releves par couche. Nomme, parce que le JSON le declare :
#: un fichier qui annonce `top: 3` doit l'avoir vraiment demande.
TOP_DEFAUT = 3

#: Precision demandee a `from_pretrained` — ecrite aussi dans le JSON.
DTYPE = "bfloat16"

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


def logit_lens(tok, mod, prompt: str, top: int = TOP_DEFAUT) -> list[dict]:
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


def ecrire(chemin, modele: str, top: int, resultats: dict) -> None:
    """Ecrit les mesures AVEC leur provenance, dans un seul fichier.

    Un fichier de mesures sans modele, sans date et sans `top` n'est pas
    verifiable : c'est ce qui est arrive au premier `sonde_gemma4b.json`, dont
    on n'a pu retrouver le modele que par le nombre de couches (35) et le nom
    du fichier. La provenance voyage donc avec les nombres.
    """
    document = {
        "_methode": {
            "sonde": "scripts/sonde_interp.py",
            "modele": modele,
            "date": datetime.date.today().isoformat(),
            "dtype": DTYPE,
            "top": top,
            "couches": max((len(c) for c in resultats.values()), default=0),
            "prompts": len(resultats),
            # Le piege de lecture principal, ecrit dans l'artefact lui-meme.
            "gabarit_conversation": False,
        },
        "prompts": resultats,
    }
    Path(chemin).write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--modele", default=DEFAUT_MODELE)
    p.add_argument("--json", help="ecrit les resultats dans ce fichier")
    args = p.parse_args(argv)

    tok, mod = charger(args.modele)
    tout = {}
    for prompt in PROMPTS:
        couches = logit_lens(tok, mod, prompt, top=TOP_DEFAUT)
        tout[prompt] = couches
        derniere = couches[-1]["top"][0]
        print(f"{prompt!r} -> sortie finale : {derniere['token']!r} "
              f"(p={derniere['p']})")
    if args.json:
        ecrire(args.json, args.modele, TOP_DEFAUT, tout)
        print(f"ecrit : {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
