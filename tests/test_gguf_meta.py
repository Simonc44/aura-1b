"""Lecture de l'architecture REELLE d'un GGUF (aura/gguf_meta.py).

Ce module existe pour repondre a une question que le nom du fichier ne peut
pas trancher : « ces adaptateurs LoRA peuvent-ils se monter sur ce cerveau ? »
Un adaptateur monte sur une autre architecture ne leve aucune erreur — il
produit du bruit. Le verdict doit donc venir du CONTENU.

Les tests qui portent sur les fichiers reels se SAUTENT si les GGUF ne sont
pas la (CI sans modeles) : un test qui echoue parce qu'un modele n'est pas
telecharge n'apprend rien a personne.
"""
from pathlib import Path

import pytest

from aura import gguf_meta

RACINE = Path(__file__).resolve().parent.parent
LLAMA = RACINE / "modeles" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
GEMMA = (RACINE / "modeles"
         / "Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_Q4_k_m.gguf")


class TestRobustesse:
    def test_fichier_absent_renvoie_vide(self):
        assert gguf_meta.architecture("inexistant.gguf") == {}
        assert gguf_meta.famille("inexistant.gguf") == ""

    def test_fichier_non_gguf_renvoie_vide(self, tmp_path):
        faux = tmp_path / "pas-un-modele.gguf"
        faux.write_bytes(b"ceci n'est pas un GGUF")
        assert gguf_meta.architecture(str(faux)) == {}

    def test_fichier_tronque_renvoie_vide(self, tmp_path):
        """Un telechargement interrompu ne doit pas planter l'appelant."""
        tronque = tmp_path / "tronque.gguf"
        if GEMMA.is_file():
            tronque.write_bytes(GEMMA.read_bytes()[:64])
        else:
            tronque.write_bytes(b"GGUF" + b"\x00" * 60)
        assert gguf_meta.architecture(str(tronque)) == {}

    def test_incompatible_ne_leve_jamais(self):
        ok, pourquoi = gguf_meta.compatible("inexistant.gguf", {"famille": "llama"})
        assert ok is False and "illisible" in pourquoi

    def test_resume_lisible_meme_sur_fichier_absent(self):
        assert "illisible" in gguf_meta.resume("inexistant.gguf")


class TestGemma:
    """Le cerveau par defaut du projet : les nombres qui decident tout."""

    @pytest.fixture(autouse=True)
    def _modele(self):
        if not GEMMA.is_file():
            pytest.skip("GGUF Gemma absent : rien a mesurer")

    def test_architecture_lue_dans_le_fichier(self):
        arc = gguf_meta.architecture(str(GEMMA))
        assert arc["famille"] == "gemma3"
        assert arc["block_count"] == 26
        assert arc["embedding_length"] == 1152
        assert arc["head_count"] == 4
        assert arc["head_count_kv"] == 1
        assert arc["context_length"] == 32768

    def test_le_nom_ne_prouve_rien(self, tmp_path, monkeypatch):
        """Copie sous un nom neutre : l'architecture reste la meme."""
        import shutil
        copie = tmp_path / "modele-au-nom-quelconque.gguf"
        try:
            shutil.copyfile(GEMMA, copie)
        except OSError:
            pytest.skip("copie impossible (espace disque)")
        assert gguf_meta.famille(str(copie)) == "gemma3"

    def test_gemma_incompatible_avec_des_lora_llama(self):
        ok, pourquoi = gguf_meta.compatible(str(GEMMA), {
            "famille": "llama", "block_count": 16, "embedding_length": 2048,
            "head_count": 32, "head_count_kv": 8})
        assert not ok
        for attendu in ("block_count", "embedding_length", "gemma3"):
            assert attendu in pourquoi


class TestLlama:
    @pytest.fixture(autouse=True)
    def _modele(self):
        if not LLAMA.is_file():
            pytest.skip("GGUF Llama absent : rien a mesurer")

    def test_architecture_lue_dans_le_fichier(self):
        arc = gguf_meta.architecture(str(LLAMA))
        assert arc["famille"] == "llama"
        assert arc["block_count"] == 16
        assert arc["embedding_length"] == 2048
        assert arc["vocab_size"] == 128256

    def test_llama_compatible_avec_ses_propres_lora(self):
        ok, pourquoi = gguf_meta.compatible(str(LLAMA), {
            "famille": "llama", "block_count": 16, "embedding_length": 2048,
            "head_count": 32, "head_count_kv": 8})
        assert ok, pourquoi


class TestPerformance:
    def test_lecture_mise_en_cache(self):
        """L'en-tete est relu a chaque appel ? Non : mis en cache."""
        if not LLAMA.is_file():
            pytest.skip("GGUF Llama absent")
        gguf_meta.architecture(str(LLAMA))
        assert str(LLAMA) in gguf_meta._CACHE
