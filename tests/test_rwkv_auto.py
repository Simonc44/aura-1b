"""Tests RWKV (mocké) et auto-amélioration."""
import os
import tempfile

from aura import autoamelioration
from aura.orchestrateur import Aura1B


class TestAutoAmelioration:
    def test_enregistrer_et_chercher(self, monkeypatch, tmp_path):
        fichier = str(tmp_path / "corr.jsonl")
        monkeypatch.setattr(autoamelioration, "_FICHIER", fichier)

        autoamelioration.enregistrer_correction(
            "quel est le carre de 5", "25", "25", "math")
        autoamelioration.enregistrer_correction(
            "qui a gagne le monde 2026", "???", "Pas encore joue", "web")

        corrections = autoamelioration.chercher_corrections_similaires(
            "quel est le carre de 7")
        assert len(corrections) >= 1
        assert "25" in corrections[0]

    def test_stats(self, monkeypatch, tmp_path):
        fichier = str(tmp_path / "corr.jsonl")
        monkeypatch.setattr(autoamelioration, "_FICHIER", fichier)
        autoamelioration.enregistrer_correction("q1", "a1", "c1", "math")
        s = autoamelioration.stats()
        assert s["total"] == 1
        assert s["par_categorie"]["math"] == 1


class TestMoE:
    def test_choisit_mamba_pour_math(self):
        ia = Aura1B(cerveau="auto")
        assert ia._choisir_cerveau({"math"}) == "mamba"

    def test_choisit_rwkv_pour_general(self):
        ia = Aura1B(cerveau="auto")
        assert ia._choisir_cerveau({"general"}) == "rwkv"

    def test_choisit_rwkv_pour_web(self):
        ia = Aura1B(cerveau="auto")
        assert ia._choisir_cerveau({"web"}) == "rwkv"

    def test_prompt_inclut_corrections(self, monkeypatch, tmp_path):
        fichier = str(tmp_path / "corr.jsonl")
        monkeypatch.setattr(autoamelioration, "_FICHIER", fichier)
        autoamelioration.enregistrer_correction(
            "test question", "mauvaise", "bonne reponse")
        prompt = Aura1B._construire_prompt("test question", "", "")
        assert "CORRECTIONS" in prompt
        assert "bonne reponse" in prompt
