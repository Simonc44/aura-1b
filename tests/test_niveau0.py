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
        # « cube de N » : meme famille que « carre de N », et le motif
        # manquait — au banc 100, ces 3 questions partaient au LLM (~40 s)
        # pendant que le carre voisin repondait en 5 ms.
        ("Quel est le cube de 3 ?", "27"),
        ("Quel est le cube de 8 ?", "512"),
        ("Calcule le cube de 5", "125"),
    ])
    def test_calculs_exact(self, question, attendu):
        assert f0.repondre(question) == attendu

    @pytest.mark.parametrize("question", [
        # « cubique » n'est pas « cube de », et une racine cubique ne doit
        # PAS etre lue comme un cube.
        "racine cubique de 27", "raconte moi une histoire de cube",
    ])
    def test_cube_ne_deborde_pas(self, question):
        assert f0.repondre(question) is None

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
        # question quasi identique -> cache hit (etage FLOU, appele seul :
        # `repondre` ne fait plus que l'EXACT depuis le 6 oct. 2026)
        r = f0.repondre_cache("Qui a ecrit Les Miserables ?")
        assert r == "Victor Hugo."
        # question differente -> None
        assert f0.repondre_cache("Explique la photosynthese") is None
        # et l'etage exact, lui, ne repond jamais sur du texte non mathematique
        assert f0.repondre("Qui a ecrit Les Miserables ?") is None

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

        f0.enregistrer("q", "[RontoGPT] Erreur de generation : x")
        assert f0.stats()["entrees_cache"] == 0

    @pytest.mark.parametrize("question", [
        "Quel est mon prenom ?", "Comment je m appelle ?", "Quel est mon age ?",
    ])
    def test_questions_contextuelles_jamais_cachees(self, monkeypatch, tmp_path,
                                                    question):
        monkeypatch.setattr(f0, "_FICHIER", tmp_path / "cache.jsonl")
        monkeypatch.setattr(f0, "_vectoriseur", None)
        monkeypatch.setattr(f0, "_matrice", None)
        monkeypatch.setattr(f0, "_entrees", [])

        # l'enregistrement est refuse
        f0.enregistrer(question, "Tu es Simon.")
        assert f0.stats()["entrees_cache"] == 0
        # la lecture ne touche jamais le cache pour ces questions
        f0.enregistrer("autre question", "reponse neutre")
        assert f0.repondre_cache(question) is None

    def test_purger_contextuelles(self, monkeypatch, tmp_path):
        monkeypatch.setattr(f0, "_vectoriseur", None)
        monkeypatch.setattr(f0, "_matrice", None)
        monkeypatch.setattr(f0, "_entrees", [])
        monkeypatch.setattr(f0, "_FICHIER", tmp_path / "cache.jsonl")

        (tmp_path / "cache.jsonl").write_text(
            '{"q": "quel est mon prenom", "r": "Tu es Simon."}\n'
            '{"q": "capitale de la france", "r": "Paris."}\n',
            encoding="utf-8")
        assert f0.purger_contextuelles() == 1
        assert f0.stats()["entrees_cache"] == 1


class TestPipelineNiveaux:
    def test_niveau0_passe_avant_llm(self):
        from aura.orchestrateur import RontoGPT
        ia = RontoGPT()
        r = ia.executer_detaille("Calcule le carre de 9")
        assert r["cerveau_choisi"] == "niveau0-instantane"
        assert r["reponse"] == "81"
