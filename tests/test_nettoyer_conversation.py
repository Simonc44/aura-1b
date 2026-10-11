"""Nettoyage de l'historique : retire les recopies de contexte, garde le reste."""
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "scripts"))

import nettoyer_conversation as nc  # noqa: E402


def _ligne(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


def test_retire_le_tour_pollue_et_sa_question():
    lignes = [
        _ligne({"role": "user", "content": "pourquoi le ciel est bleu"}),
        _ligne({"role": "assistant",
                "content": "Voici une réponse ... la diffusion de Rayleigh ..."}),
        _ligne({"role": "user", "content": "ok"}),
    ]
    gardees, retires = nc.nettoyer(lignes)
    assert retires == 1
    assert [json.loads(g)["content"] for g in gardees] == ["ok"]


def test_garde_une_reponse_saine():
    lignes = [_ligne({"role": "user", "content": "q"}),
              _ligne({"role": "assistant", "content": "42."})]
    gardees, retires = nc.nettoyer(lignes)
    assert retires == 0 and gardees == lignes


def test_ne_detruit_pas_une_ligne_illisible():
    lignes = ["pas du json", _ligne({"role": "assistant", "content": "ok"})]
    gardees, retires = nc.nettoyer(lignes)
    assert retires == 0 and gardees == lignes


def test_reconnait_les_consignes_doutil():
    assert nc.est_polluee("Pour lire un fichier texte de l'ordinateur, ...")
    assert not nc.est_polluee("La photosynthèse convertit la lumière.")
