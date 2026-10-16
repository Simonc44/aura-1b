"""La sonde publie sa provenance, pas seulement ses nombres.

`torch` n'est pas une dependance du projet : ce fichier ne teste donc QUE
l'ecriture du JSON (aucun modele charge). C'est justement le point qui compte
pour un resultat publie — le `sonde_gemma4b.json` du 16 oct. 2026 ne portait ni
modele, ni date, ni `top`, et son modele n'a pu etre retrouve que par le nombre
de couches (35 = 34 couches + l'embedding) et le nom du fichier.
"""
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "scripts"))

import sonde_interp as si  # noqa: E402

_MESURES = {
    "Q: Combien font 17 * 23 ?\nR:": [
        {"couche": 0, "top": [{"token": ":", "p": 1.0}]},
        {"couche": 1, "top": [{"token": " 391", "p": 0.5},
                              {"token": " 3911", "p": 0.25},
                              {"token": " 39", "p": 0.125}]},
    ],
}


def test_le_json_porte_sa_provenance(tmp_path):
    chemin = tmp_path / "sonde.json"
    si.ecrire(chemin, "unsloth/gemma-3-4b-it", 3, _MESURES)

    methode = json.loads(chemin.read_text(encoding="utf-8"))["_methode"]
    assert methode["modele"] == "unsloth/gemma-3-4b-it"
    assert methode["top"] == 3
    assert methode["dtype"] == "bfloat16"
    assert methode["couches"] == 2, "le nombre de positions, pas des prompts"
    assert methode["prompts"] == 1
    assert len(methode["date"]) == 10, "une date ISO, pas l'horodatage du fichier"
    # Le piege mesure : le modele est sonde SANS son gabarit de conversation,
    # donc hors de son format d'entrainement. Il est ecrit dans l'artefact.
    assert methode["gabarit_conversation"] is False


def test_les_mesures_sont_recopiees_sans_etre_touchees(tmp_path):
    chemin = tmp_path / "sonde.json"
    si.ecrire(chemin, "unsloth/gemma-3-4b-it", si.TOP_DEFAUT, _MESURES)
    assert json.loads(chemin.read_text(encoding="utf-8"))["prompts"] == _MESURES
