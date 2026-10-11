"""Parite d'echantillonnage entre les DEUX chemins d'inference.

Defaut reel : les parametres etaient ecrits DEUX FOIS (une fois pour le
serveur llama.cpp, une fois pour le chemin in-process). Ils avaient deja
divergé — le serveur, qui est le chemin ACTIF, n'envoyait aucune penalite et
n'avait pas de relance anti-boucle.

Deux verrous ici :
  1. les profils partages donnent les MEMES valeurs aux deux backends ;
  2. plus aucune valeur de penalite n'est ecrite en dur hors de
     `aura/echantillonnage.py` — c'est ce controle qui empeche la divergence
     de revenir par une petite modification locale bien intentionnee.
"""
from pathlib import Path

import pytest

from aura import echantillonnage as E

_AURA = Path(__file__).resolve().parent.parent / "aura"

#: cles communes aux deux backends (le DRY est server-only, par nature)
_CLES = ("temperature", "top_p", "min_p", "repeat_penalty")


@pytest.mark.parametrize("profil", [E.NORMAL, E.RELANCE])
def test_memes_valeurs_des_deux_cotes(profil):
    serveur = E.pour_serveur(profil)
    inproc = E.pour_inprocess(profil)
    for cle in _CLES:
        assert serveur[cle] == inproc[cle], (
            f"{cle} diverge entre les chemins pour {profil}")


def test_dry_uniquement_cote_serveur():
    """Le sampler DRY n'existe pas cote llama-cpp-python : l'ecart est declare."""
    assert "dry_multiplier" in E.pour_serveur(E.NORMAL)
    assert "dry_multiplier" not in E.pour_inprocess(E.NORMAL)
    # ...et il est desactivable (retour au comportement d'origine du serveur)
    assert "dry_multiplier" not in E.pour_serveur(E.NORMAL, sans_dry=True)


def test_balises_darret_suivent_la_famille_du_cerveau():
    """Le stop generique n'impose plus une famille.

    `<|eot_id|>` (Llama) etait ecrit en dur dans STOPS : avec le cerveau
    Gemma, on envoyait au modele une balise absente de son vocabulaire, sans
    que rien ne le signale. Les balises de la famille REELLEMENT chargee sont
    ajoutees par `stops()`, lues dans le GGUF.
    """
    balises = E.stops()
    from aura import llama_cerveau
    famille = llama_cerveau.famille()
    if famille.startswith("gemma"):
        assert "<end_of_turn>" in balises
        assert "<|eot_id|>" not in balises
    elif famille == "llama":
        assert "<|eot_id|>" in balises
    # les gardes generiques restent dans tous les cas
    assert "\nUtilisateur:" in balises and len(balises) == len(set(balises))


def test_aucune_valeur_codee_en_dur_hors_du_module():
    """Ni `serveur_lora` ni `llama_cerveau` ne fixent une penalite."""
    interdits = ("repeat_penalty", "min_p", "top_p")
    for nom in ("serveur_lora.py", "llama_cerveau.py"):
        for i, ligne in enumerate((_AURA / nom).read_text(
                encoding="utf-8").splitlines(), 1):
            nue = ligne.split("#", 1)[0]
            if not any(m in nue for m in interdits):
                continue
            # une VALEUR litterale ("repeat_penalty": 1.1 / min_p=0.05) est
            # interdite ; lire un attribut de profil (p.repeat_penalty) non.
            for mot in interdits:
                assert f'"{mot}":' not in nue and f"{mot}=" not in nue, (
                    f"{nom}:{i} fixe {mot} en dur — mettre la valeur dans "
                    f"aura/echantillonnage.py : {ligne.strip()}")
