"""Tests du Niveau 0 : filtre instantane (calculs exacts + cache semantique)."""
import pytest

from aura import filtre_instantane as f0


class TestCalculDirect:
    @pytest.mark.parametrize("question,attendu", [
        ("Calcule le carre de 12", "144"),
        ("2 puissance 10", "1 024"),
        ("15% de 200", "30"),
        ("combien fait 45*12+3", "543"),
        ("racine carree de 144", "12"),
        ("100-25", "75"),
    ])
    def test_calculs_exact(self, question, attendu):
        assert f0.repondre(question) == attendu

    @pytest.mark.parametrize("question", [
        "bonjour", "Qui a ecrit Victor Hugo ?", "", "aaa bbb ccc",
    ])
    def test_non_math_renvoie_none(self, question, monkeypatch, tmp_path):
        # cache isole : sans ca, une vraie entree du disque ferait un hit
        monkeypatch.setattr(f0, "_FICHIER", tmp_path / "cache.jsonl")
        monkeypatch.setattr(f0, "_vectoriseur", None)
        monkeypatch.setattr(f0, "_matrice", None)
        monkeypatch.setattr(f0, "_entrees", [])
        assert f0.repondre(question) is None

    def test_securite_ast(self):
        # eval() classique executerait ca ; l'AST refuse
        assert f0.repondre("__import__('os').system('dir')") is None

    def test_puissance_garde_fou(self):
        # 2 puissance 100000 serait legitime mais explosif : refuse -> None
        assert f0.repondre("2 puissance 100000") is None


class TestCacheSemantique:
    def test_enregistrer_et_retrouver(self, monkeypatch, tmp_path):
        monkeypatch.setattr(f0, "_FICHIER", tmp_path / "cache.jsonl")
        monkeypatch.setattr(f0, "_vectoriseur", None)
        monkeypatch.setattr(f0, "_matrice", None)
        monkeypatch.setattr(f0, "_entrees", [])

        f0.enregistrer("Qui a ecrit Les Miserables ?", "Victor Hugo.")
        # question quasi identique -> cache hit
        r = f0.repondre("Qui a ecrit Les Miserables ?")
        assert r == "Victor Hugo."
        # question differente -> None
        assert f0.repondre("Explique la photosynthese") is None

    def test_pas_de_doublon(self, monkeypatch, tmp_path):
        monkeypatch.setattr(f0, "_FICHIER", tmp_path / "cache.jsonl")
        monkeypatch.setattr(f0, "_vectoriseur", None)
        monkeypatch.setattr(f0, "_matrice", None)
        monkeypatch.setattr(f0, "_entrees", [])

        f0.enregistrer("question test", "reponse A")
        f0.enregistrer("question test", "reponse B")
        assert f0.stats()["entrees_cache"] == 1

    def test_ne_pas_enregistrer_erreurs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(f0, "_FICHIER", tmp_path / "cache.jsonl")
        monkeypatch.setattr(f0, "_vectoriseur", None)
        monkeypatch.setattr(f0, "_matrice", None)
        monkeypatch.setattr(f0, "_entrees", [])

        f0.enregistrer("q", "[Aura] Erreur de generation : x")
        assert f0.stats()["entrees_cache"] == 0


class TestPipelineNiveaux:
    def test_niveau0_passe_avant_llm(self):
        from aura.orchestrateur import Aura1B
        ia = Aura1B()
        r = ia.executer_detaille("Calcule le carre de 9")
        assert r["cerveau_choisi"] == "niveau0-instantane"
        assert r["reponse"] == "81"
