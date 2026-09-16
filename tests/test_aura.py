"""Tests Aura-1B : cerveau Llama 3.2 1B + experts symboliques."""
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


class TestLlamaCerveau:
    def test_gguf_present(self):
        from aura import llama_cerveau
        # Le fichier GGUF doit etre telecharge (sinon le test saute proprement)
        if not llama_cerveau.disponible():
            import pytest
            pytest.skip("GGUF Llama non telecharge (scripts/telecharger_llama.py)")

    def test_reponse_fr(self):
        from aura import llama_cerveau
        if not llama_cerveau.disponible():
            import pytest
            pytest.skip("GGUF Llama non telecharge")
        r = llama_cerveau.generer("Quelle est la capitale de la France ?",
                                  max_tokens=30)
        assert "Paris" in r

    def test_repli_sans_gguf(self, monkeypatch, tmp_path):
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "_CHEMIN_GGUF",
                            str(tmp_path / "inexistant.gguf"))
        monkeypatch.setattr(llama_cerveau, "_llm", None)
        r = llama_cerveau.generer("test")
        assert "pas disponible" in r or "introuvable" in r


class TestOrchestrateur:
    def test_analyse_question(self):
        r = Aura1B().analyser("calcule 3 puissance 4")
        assert "math" in r["experts"]

    def test_cerveau_est_llama(self, monkeypatch):
        # neutralise le niveau 0 (cache disque partagé) : on teste le repli LLM
        from aura import filtre_instantane
        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
        r = Aura1B().executer_detaille("bonjour")
        assert r["cerveau_choisi"] == "llama-3.2-1b"

    def test_prompt_inclut_corrections(self, monkeypatch, tmp_path):
        f = str(tmp_path / "c.jsonl")
        monkeypatch.setattr(autoamelioration, "_FICHIER", f)
        autoamelioration.enregistrer_correction("test", "mauvaise", "bonne")
        p = Aura1B._construire_prompt("test", "", "")
        assert "CORRECTIONS" in p and "bonne" in p

    def test_prompt_inclut_web_et_formule(self):
        p = Aura1B._construire_prompt("q", "fait web", "mul(X0,X0)")
        assert "WEB" in p and "FORMULE" in p
