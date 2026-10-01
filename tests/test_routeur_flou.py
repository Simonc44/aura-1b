"""Classificateur flou (brique 2) : 2e avis du routeur sur vecteurs MiniLM.

Le routeur historique classe sur des n-grams (FORMULES) ; celui-ci classe
sur le SENS (vecteurs 384-d, PARAPHRASES). Les deux avis sont fondus
50/50 dans RouteurIntelligent.classer.
"""
import pytest

from aura import embeddings, routeur

_moder = pytest.mark.skipif(
    not embeddings.modele_present(),
    reason="modeles/minilm absent : scripts/telecharger_minilm.py")


class TestExemplesPartages:
    """exemples_entrainement() = source de verite unique des deux lecteurs."""

    def test_coherent(self):
        questions, labels = routeur.exemples_entrainement()
        assert len(questions) == len(labels) >= 60
        assert {"math", "web", "general"} <= set(labels)

    def test_savoir_de_base_inclus(self):
        questions, _labels = routeur.exemples_entrainement()
        assert routeur._EXEMPLES_MATH[0] in questions
        assert routeur._EXEMPLES_WEB[0] in questions
        assert routeur._EXEMPLES_GENERAL[0] in questions


class TestCoupe:
    def test_aura_flo_zero(self, monkeypatch):
        monkeypatch.setenv("AURA_FLOU", "0")
        from aura import classificateur_flou as cf
        assert cf.actif() is False
        assert cf.probas("explique moi ce concept") is None
        # le routeur retombe sur son avis TF-IDF historique
        assert routeur._probas_flou("explique moi ce concept") is None

    def test_routeur_sans_flou_inchange(self, monkeypatch):
        monkeypatch.setenv("AURA_FLOU", "0")
        d = routeur.RouteurIntelligent().classer("calcule 2 puissance 10")
        assert "MiniLM-SVM" not in d["methodes"]
        assert d["prediction"] == "math"
        assert d["scores"]


class TestFusion:
    def test_fusion_50_50_renormalisee(self, monkeypatch):
        # avis flou force vers « math » : les scores doivent bouger
        monkeypatch.setattr(routeur, "_probas_flou",
                            lambda q: {"math": 1.0, "web": 0.0,
                                       "general": 0.0})
        d = routeur.RouteurIntelligent().classer("explique moi ce concept")
        assert "MiniLM-SVM" in d["methodes"]
        assert abs(sum(d["scores"].values()) - 1.0) < 0.01

        # meme question, sans avis flou : scores different, pas de marque
        monkeypatch.setattr(routeur, "_probas_flou", lambda q: None)
        d2 = routeur.RouteurIntelligent().classer("explique moi ce concept")
        assert "MiniLM-SVM" not in d2["methodes"]
        assert d["scores"] != d2["scores"]

    def test_echec_flou_ne_casse_rien(self, monkeypatch):
        # None (indisponible) -> simple repli, aucune exception
        monkeypatch.setattr(routeur, "_probas_flou", lambda q: None)
        d = routeur.RouteurIntelligent().classer("quel temps fait il")
        assert d["prediction"] in ("math", "web", "general")


@_moder
class TestAvisFlouReel:
    def test_probas_bien_formees(self):
        from aura import classificateur_flou as cf
        p = cf.probas("explique moi ce concept de philosophie")
        assert p is not None
        assert set(p) == {"math", "web", "general"}
        assert abs(sum(p.values()) - 1.0) < 0.02   # softmax

    def test_fusion_reelle(self):
        d = routeur.RouteurIntelligent().classer("explique moi ce concept")
        assert "MiniLM-SVM" in d["methodes"]
        assert d["prediction"] == "general"
