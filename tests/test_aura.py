"""Tests Aura-1B MoE autonome (sans Ollama)."""
import numpy as np

from aura import expert_symbolique, memoire_web, routeur, autoamelioration
from aura.orchestrateur import Aura1B


class TestExpertSymbolique:
    def test_decouvre_cube(self):
        ex = expert_symbolique.ExpertSymbolique(population=1500, generations=20)
        X = [[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]]
        y = [0.0, 0.008, 0.064, 0.216, 0.512, 1.0]
        formule = ex.resoudre(X, y)
        assert formule and ex.erreur < 1e-6

    def test_decouvre_lineaire(self):
        ex = expert_symbolique.ExpertSymbolique(population=800, generations=12)
        formule = ex.resoudre([[1.0], [2.0], [3.0], [4.0]], [3.0, 6.0, 9.0, 12.0])
        assert formule and ex.erreur < 1e-6

    def test_pow_protege(self):
        from aura.expert_symbolique import _pow_protege
        assert np.all(np.isfinite(_pow_protege(
            np.array([0.0, -2.0, 1000.0]),
            np.array([100.0, 3.0, -99.0]))))

    def test_trop_peu(self):
        assert expert_symbolique.ExpertSymbolique().resoudre([[1.0], [2.0]], [1.0, 4.0]) == ""


class TestRouteur:
    def test_detecte_math(self):
        assert "math" in routeur.RouteurIntelligent().classer("calcule 2 puissance 10")["experts"]

    def test_detecte_web(self):
        assert "web" in routeur.RouteurIntelligent().classer("qui a gagne le monde 2026")["experts"]

    def test_detecte_general(self):
        assert "general" in routeur.RouteurIntelligent().classer("bonjour")["experts"]


class TestAutoAmelioration:
    def test_enregistrer_chercher(self, monkeypatch, tmp_path):
        f = str(tmp_path / "c.jsonl")
        monkeypatch.setattr(autoamelioration, "_FICHIER", f)
        autoamelioration.enregistrer_correction("carre de 5", "20", "25", "math")
        c = autoamelioration.chercher_corrections_similaires("carre de 7")
        assert len(c) >= 1 and "25" in c[0]


class TestMoE:
    def test_choisit_mamba_pour_math(self):
        assert Aura1B()._choisir_cerveau({"math"}) == "mamba"

    def test_choisit_rwkv_pour_general(self):
        assert Aura1B()._choisir_cerveau({"general"}) == "rwkv"

    def test_analyse_question(self):
        r = Aura1B().analyser("calcule 3 puissance 4")
        assert "math" in r["experts"]

    def test_prompt_inclut_corrections(self, monkeypatch, tmp_path):
        f = str(tmp_path / "c.jsonl")
        monkeypatch.setattr(autoamelioration, "_FICHIER", f)
        autoamelioration.enregistrer_correction("test", "mauvaise", "bonne")
        p = Aura1B._construire_prompt("test", "", "")
        assert "CORRECTIONS" in p and "bonne" in p
