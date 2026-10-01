"""Memoire semantique (brique 3) : index vectoriel plat NumPy.

Deux lectures du cache de reponses (aura/.cache_reponses.jsonl) :
1. vecteurs MiniLM (cosinus >= 0.70) : une reformulation est reconnue,
   index dense (n,384) persiste en .vecs.npy (recharge ~20 ms, sans
   re-encodage) ;
2. repli TF-IDF char n-grams (>= 0.85) : present sans MiniLM.
Le garde-fou _hit_douteux (jetons porteurs) filtre les faux hits en aval.
"""
from pathlib import Path

import numpy as np
import pytest

from aura import embeddings, filtre_instantane as f0

_moder = pytest.mark.skipif(
    not embeddings.modele_present(),
    reason="modeles/minilm absent : scripts/telecharger_minilm.py")

# reformulation : cosinus MiniLM ~0.73 (hit) mais TF-IDF seul 0.71 (rate)
_Q_LONGUE = ("conjugue parler au present a la premiere personne du singulier")
_Q_COURTE = "conjugue parler au present"


def _isole(monkeypatch, tmp_path):
    """Cache neuf : aucun hit du disque reel ne peut tricher."""
    fichier = tmp_path / "cache.jsonl"
    monkeypatch.setattr(f0, "_FICHIER", fichier)
    monkeypatch.setattr(f0, "_vectoriseur", None)
    monkeypatch.setattr(f0, "_matrice", None)
    monkeypatch.setattr(f0, "_entrees", [])
    monkeypatch.setattr(f0, "_vecteurs", None)
    return fichier


class TestRepliTfidf:
    def test_hit_exact_sans_vecteur(self, monkeypatch, tmp_path):
        _isole(monkeypatch, tmp_path)
        monkeypatch.setenv("AURA_MINILM", "0")
        f0.enregistrer("Qui a ecrit Les Miserables ?", "Victor Hugo.")
        assert f0._vecteurs is None          # index coupe
        assert f0.repondre("Qui a ecrit Les Miserables ?") == "Victor Hugo."
        assert f0.repondre("Explique la photosynthese") is None

    def test_sans_modele_rien_ne_casse(self, monkeypatch, tmp_path):
        # MiniLM absent (encoder -> None) : comportement historique exact
        _isole(monkeypatch, tmp_path)
        monkeypatch.setattr(embeddings, "encoder", lambda textes: None)
        monkeypatch.setattr(embeddings, "encoder1", lambda texte: None)
        f0.enregistrer("Qui a ecrit Les Miserables ?", "Victor Hugo.")
        assert f0._vecteurs is None
        assert f0.repondre("Qui a ecrit Les Miserables ?") == "Victor Hugo."


@_moder
class TestIndexSemantique:
    def test_npy_cree_puis_recharge(self, monkeypatch, tmp_path):
        fichier = _isole(monkeypatch, tmp_path)
        f0.enregistrer(_Q_LONGUE, "Je parle.")
        npy = Path(str(fichier) + ".vecs.npy")
        assert npy.exists()
        assert np.load(npy).shape == (1, 384)

        # recharge a froid : le .npy est relu, pas re-encode
        monkeypatch.setattr(f0, "_vectoriseur", None)
        monkeypatch.setattr(f0, "_matrice", None)
        monkeypatch.setattr(f0, "_entrees", [])
        monkeypatch.setattr(f0, "_vecteurs", None)
        f0._charger_cache()
        assert f0._vecteurs is not None
        assert f0._vecteurs.shape == (1, 384)

    def test_hit_semantique_que_le_tfidf_rate(self, monkeypatch, tmp_path):
        fichier = _isole(monkeypatch, tmp_path)
        f0.enregistrer(_Q_LONGUE, "Je parle.")
        # 1) le vecteur trouve la reformulation
        assert f0.repondre(_Q_COURTE) == "Je parle."

        # 2) preuve que le TF-IDF seul ne l'aurait PAS trouvee
        monkeypatch.setenv("AURA_MINILM", "0")
        for attr in ("_vectoriseur", "_matrice", "_entrees", "_vecteurs"):
            monkeypatch.setattr(f0, attr, None if attr != "_entrees" else [])
        f0._charger_cache()
        assert f0.repondre(_Q_COURTE) is None

    def test_garde_fou_verbe_different(self, monkeypatch, tmp_path):
        # cosinus MiniLM ~0.72 (au-dessus du seuil) mais « pouvoir » est
        # absent du cache : _hit_douteux refuse (bug historique des
        # conjugaisons croisees)
        _isole(monkeypatch, tmp_path)
        f0.enregistrer(_Q_LONGUE, "Je parle.")
        assert f0.repondre("conjugue pouvoir au present") is None

    def test_purger_remet_l_index_a_plat(self, monkeypatch, tmp_path):
        fichier = _isole(monkeypatch, tmp_path)
        fichier.write_text(
            '{"q": "quel est mon prenom", "r": "Tu es Simon."}\n'
            '{"q": "capitale de la france", "r": "Paris."}\n',
            encoding="utf-8")
        assert f0.purger_contextuelles() == 1
        # l'index vectoriel deplace est supprime avec les entrees retirees
        assert not Path(str(fichier) + ".vecs.npy").exists()
        # reconstruit au prochain acces : une seule entree restante
        f0.stats()
        assert f0._vecteurs is not None
        assert np.load(str(fichier) + ".vecs.npy").shape == (1, 384)
