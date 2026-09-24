"""Tests de la rotation du secret AEF (scripts/tourner_secret.py).

Petits conteneurs AUAE synthetiques (payloads de quelques octets) :
on teste la LOGIQUE (verifications, round-trip, remplacement atomique,
echecs propres) sans jamais manipuler les 808 Mo reels.
"""
import hashlib

import pytest

from aura import chiffrement

MODULE_PATH = "scripts.tourner_secret"
tourner_secret = pytest.importorskip("scripts.tourner_secret")
tourner = tourner_secret.tourner
verifier = tourner_secret.verifier


@pytest.fixture()
def atelier(tmp_path):
    """Un .enc synthetique + son secret, dans un dossier isole."""
    enc = tmp_path / "aura_system.aef.enc"
    secret = tmp_path / ".secret_aef.txt"
    clair = b"AURA" + b"x" * 4096            # faux .aef (magic requis)
    secret.write_text("ancien-secret-123\n", encoding="utf-8")
    enc.write_bytes(chiffrement.chiffrer(clair, "ancien-secret-123"))
    return {"enc": enc, "secret": secret, "clair": clair,
            "sha": hashlib.sha256(clair).hexdigest()}


class TestVerifier:
    def test_verifie_le_vrai_secret(self, atelier):
        r = verifier(atelier["enc"], atelier["secret"])
        assert r["ok"] is True
        assert r["sha"] == atelier["sha"]

    def test_echec_propre_mauvais_secret(self, atelier):
        atelier["secret"].write_text("mauvais\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            verifier(atelier["enc"], atelier["secret"])
        # rien n'a bouge
        assert atelier["enc"].exists()

    def test_refuse_non_auae(self, atelier):
        atelier["enc"].write_bytes(b"XXXX" + b"y" * 100)
        with pytest.raises(SystemExit):
            verifier(atelier["enc"], atelier["secret"])


class TestTourner:
    def test_rotation_complete_round_trip(self, atelier):
        r = tourner(atelier["enc"], atelier["secret"],
                    nouveau="nouveau-secret-456", confirmer=False)
        assert r["sha"] == atelier["sha"]
        # le nouveau secret dechiffre, l'ancien non
        clair = chiffrement.dechiffrer(atelier["enc"].read_bytes(),
                                       "nouveau-secret-456")
        assert hashlib.sha256(clair).hexdigest() == atelier["sha"]
        with pytest.raises(ValueError):
            chiffrement.dechiffrer(atelier["enc"].read_bytes(),
                                   "ancien-secret-123")
        # le fichier secret contient le nouveau secret
        assert atelier["secret"].read_text(encoding="utf-8").strip() \
            == "nouveau-secret-456"
        # .bak crees
        assert atelier["enc"].with_suffix(".enc.bak").exists()
        assert atelier["secret"].with_suffix(".txt.bak").exists()

    def test_echec_ancien_secret_ne_dechiffre_plus(self, atelier):
        # le .enc a ete chiffre avec un autre secret que .secret_aef.txt
        atelier["enc"].write_bytes(
            chiffrement.chiffrer(atelier["clair"], "autre-secret"))
        with pytest.raises(SystemExit):
            tourner(atelier["enc"], atelier["secret"],
                    nouveau="x", confirmer=False)
        # rien n'a bouge : le secret d'origine est intact
        assert atelier["secret"].read_text(encoding="utf-8").strip() \
            == "ancien-secret-123"

    def test_abandon_si_confirmation_refusee(self, atelier, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *a: "NON")
        with pytest.raises(SystemExit):
            tourner(atelier["enc"], atelier["secret"], confirmer=True)
        assert atelier["secret"].read_text(encoding="utf-8").strip() \
            == "ancien-secret-123"
        assert chiffrement.dechiffrer(atelier["enc"].read_bytes(),
                                      "ancien-secret-123") == atelier["clair"]
        assert not atelier["enc"].with_suffix(".enc.tmp").exists()

    def test_refuse_secret_identique(self, atelier):
        with pytest.raises(SystemExit):
            tourner(atelier["enc"], atelier["secret"],
                    nouveau="ancien-secret-123", confirmer=False)

    def test_tmp_nettoye_apres_succes(self, atelier):
        tourner(atelier["enc"], atelier["secret"],
                nouveau="s3", confirmer=False)
        assert not atelier["enc"].with_suffix(".enc.tmp").exists()
