"""Tests du pipeline Aura-1B (sans reseau, sans LLM)."""
import numpy as np

from aura import expert_symbolique, memoire_web
from aura.orchestrateur import Aura1B


class TestExpertSymbolique:
    def test_decouvre_cube(self):
        ex = expert_symbolique.ExpertSymbolique(population=1500, generations=20)
        # domaine unite : la loi normalisee est exactement X0^3, representable sans constante
        X = [[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]]
        y = [0.0, 0.008, 0.064, 0.216, 0.512, 1.0]
        formule = ex.resoudre(X, y)
        assert formule, "aucune formule trouvee"
        assert ex.erreur is not None and ex.erreur < 1e-6, f"erreur trop grande : {ex.erreur}"

    def test_decouvre_lineaire_exactement(self):
        ex = expert_symbolique.ExpertSymbolique(population=800, generations=12)
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [3.0, 6.0, 9.0, 12.0]  # Y = 3X -> normalisee : Y' = X'
        formule = ex.resoudre(X, y)
        assert formule
        assert ex.erreur < 1e-6

    def test_pow_protege_ne_plante_pas(self):
        from aura.expert_symbolique import _pow_protege
        x1 = np.array([0.0, -2.0, 1000.0])
        x2 = np.array([100.0, 3.0, -99.0])
        r = _pow_protege(x1, x2)
        assert np.all(np.isfinite(r)), "pow protege doit toujours renvoyer du fini"

    def test_trop_peu_d_exemples(self):
        ex = expert_symbolique.ExpertSymbolique()
        assert ex.resoudre([[1.0], [2.0]], [1.0, 4.0]) == ""


class TestMemoireWeb:
    def test_routeur_detecte_besoin_web(self):
        assert memoire_web.a_besoin_web("Qui a gagne le dernier match ?")
        assert memoire_web.a_besoin_web("actualite 2026")
        assert not memoire_web.a_besoin_web("resous x^3 pour x=2")

    def test_recherche_offline_renvoie_vide(self):
        # sans assurance reseau : la fonction ne doit JAMAIS lever
        r = memoire_web.chercher("question test xyzzy", max_resultats=1, timeout=3)
        assert isinstance(r, str)


class TestOrchestrateur:
    def test_fallback_sans_llm(self):
        ia = Aura1B(hote="http://127.0.0.1:1", timeout=1)  # port invalide volontaire
        r = ia.executer_detaille("resous x^3", [[2.0], [3.0], [4.0]], [8.0, 27.0, 64.0])
        assert "formule" in r and r["formule"] != ""
        assert "Question" in r["reponse"], "le fallback doit rester honnete et structure"

    def test_routage_sans_donnees(self):
        ia = Aura1B(hote="http://127.0.0.1:1", timeout=1)
        r = ia.executer_detaille("bonjour")
        assert r["web_actif"] is False
        assert r["formule"] == ""

    def test_repli_mamba_vers_ollama(self, monkeypatch):
        """Si Mamba est indisponible, Aura doit replier sur Ollama puis sur le template."""
        import pytest

        from aura import mamba_orchestrateur

        class _MambaCassé:
            def __init__(self, *a, **k):
                raise mamba_orchestrateur.MambaIndisponible("test simulé")

        monkeypatch.setattr(mamba_orchestrateur, "OrchestrateurMamba", _MambaCassé)
        ia = Aura1B(hote="http://127.0.0.1:1", timeout=1, cerveau="mamba")
        r = ia.executer_detaille("resous x^3", [[2.0], [3.0], [4.0]], [8.0, 27.0, 64.0])
        assert r["formule"] != "", "le pipeline doit continuer malgré l'échec Mamba"
        assert "Question" in r["reponse"]
