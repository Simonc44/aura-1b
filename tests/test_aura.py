"""Tests du pipeline Aura-1B (sans reseau, sans LLM)."""
import numpy as np

from aura import expert_symbolique, memoire_web, routeur
from aura.orchestrateur import Aura1B


class TestExpertSymbolique:
    def test_decouvre_cube(self):
        ex = expert_symbolique.ExpertSymbolique(population=1500, generations=20)
        X = [[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]]
        y = [0.0, 0.008, 0.064, 0.216, 0.512, 1.0]
        formule = ex.resoudre(X, y)
        assert formule, "aucune formule trouvee"
        assert ex.erreur is not None and ex.erreur < 1e-6

    def test_decouvre_lineaire_exactement(self):
        ex = expert_symbolique.ExpertSymbolique(population=800, generations=12)
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [3.0, 6.0, 9.0, 12.0]
        formule = ex.resoudre(X, y)
        assert formule
        assert ex.erreur < 1e-6

    def test_pow_protege_ne_plante_pas(self):
        from aura.expert_symbolique import _pow_protege
        x1 = np.array([0.0, -2.0, 1000.0])
        x2 = np.array([100.0, 3.0, -99.0])
        r = _pow_protege(x1, x2)
        assert np.all(np.isfinite(r))

    def test_trop_peu_d_exemples(self):
        ex = expert_symbolique.ExpertSymbolique()
        assert ex.resoudre([[1.0], [2.0]], [1.0, 4.0]) == ""


class TestRouteurIntelligent:
    def test_detecte_math(self):
        r = routeur.RouteurIntelligent()
        res = r.classer("calcule 2 puissance 10")
        assert "math" in res["experts"]

    def test_detecte_web(self):
        r = routeur.RouteurIntelligent()
        res = r.classer("qui a gagne la coupe du monde 2026")
        assert "web" in res["experts"]

    def test_detecte_general(self):
        r = routeur.RouteurIntelligent()
        res = r.classer("bonjour comment ca va")
        assert "general" in res["experts"]

    def test_routeur_classique(self):
        r = routeur.RouteurIntelligent()
        res = r.routeur_classique("calcule cette equation")
        assert "math" in res["experts"]


class TestMemoireWeb:
    def test_routeur_detecte_besoin_web(self):
        assert memoire_web.a_besoin_web("Qui a gagne le dernier match ?")
        assert memoire_web.a_besoin_web("actualite 2026")
        assert not memoire_web.a_besoin_web("resous x^3 pour x=2")

    def test_recherche_offline_renvoie_vide(self):
        r = memoire_web.chercher("question test xyzzy", max_resultats=1, timeout=3)
        assert isinstance(r, str)


class TestOrchestrateur:
    def test_analyse_question_math(self):
        ia = Aura1B(hote="http://127.0.0.1:1", timeout=1)
        res = ia.analyser("calcule 3 puissance 4")
        assert "math" in res["experts"]

    def test_analyse_question_web(self):
        ia = Aura1B(hote="http://127.0.0.1:1", timeout=1)
        res = ia.analyser("quel est le prix du bitcoin")
        assert "web" in res["experts"]

    def test_fallback_sans_llm(self):
        ia = Aura1B(hote="http://127.0.0.1:1", timeout=1)
        r = ia.executer_detaille("resous x^3", [[2.0], [3.0], [4.0]], [8.0, 27.0, 64.0])
        assert r["formule"] != ""
        assert "Question" in r["reponse"]

    def test_repli_mamba_vers_ollama(self, monkeypatch):
        from aura import mamba_orchestrateur

        class _MambaCasse:
            def __init__(self, *a, **k):
                raise mamba_orchestrateur.MambaIndisponible("test simule")

        monkeypatch.setattr(mamba_orchestrateur, "OrchestrateurMamba", _MambaCasse)
        ia = Aura1B(hote="http://127.0.0.1:1", timeout=1, cerveau="mamba")
        r = ia.executer_detaille("resous x^3", [[2.0], [3.0], [4.0]], [8.0, 27.0, 64.0])
        assert r["formule"] != ""
        assert "Question" in r["reponse"]
