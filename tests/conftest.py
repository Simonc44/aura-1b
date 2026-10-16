"""Fixtures partagees : mini .aef forge rapide (faux GGUF de 1 Mo)."""
import pytest


@pytest.fixture(autouse=True)
def _apis_publiques_coupees(monkeypatch):
    """Aucun reseau : les API publiques sont coupees dans TOUS les tests.

    Les tests dedieses de aura/apis_publiques.py re-active la brique avec
    monkeypatch.setenv("AURA_APIS", "1") + un urllib mocke.
    """
    monkeypatch.setenv("AURA_APIS", "0")


@pytest.fixture(autouse=True)
def _gemini_coupe(monkeypatch):
    """Aucun appel reseau vers Google, meme si ta cle est dans ton shell/.env.

    Le repli Gemini (aura/gemini.py) s'active des que GOOGLE_API_KEY (ou
    GEMINI_API_KEY) existe — y compris dans le `.env` de la racine. Sans cette
    fixture, les tests qui font rendre une reponse locale VIDE l'appelleraient
    pour de vrai. Les tests de aura/gemini.py re-activent la brique avec
    monkeypatch.setenv("AURA_GEMINI", "1") + un urllib mocke.
    """
    monkeypatch.setenv("AURA_GEMINI", "0")


@pytest.fixture(autouse=True)
def _cache_reponses_isole(monkeypatch, tmp_path):
    """Cache de reponses ISOLE pour TOUS les tests.

    Sans ca, `filtre_instantane` lit `aura/.cache_reponses.jsonl` — la
    memoire REELLE du projet. Un hit du disque y ferait passer un test qui
    repose precisement sur le fait que rien ne repond.

    C'etait latent jusqu'au 6 oct. 2026 : les tests neutralisaient l'etage
    complet en patchant `filtre_instantane.repondre`. Depuis que le cache
    est un etage SEPARE (`repondre_cache`), ce patch ne suffit plus — il
    faut isoler au niveau du conftest, pas test par test.
    """
    from aura import filtre_instantane as f0
    monkeypatch.setattr(f0, "_FICHIER", tmp_path / "cache_tests.jsonl")
    monkeypatch.setattr(f0, "_vectoriseur", None)
    monkeypatch.setattr(f0, "_matrice", None)
    monkeypatch.setattr(f0, "_entrees", [])
    monkeypatch.setattr(f0, "_vecteurs", None)


@pytest.fixture(scope="session")
def aef_forge(tmp_path_factory):
    from scripts.forger_aef import forger
    tmp = tmp_path_factory.mktemp("aef")
    faux_gguf = tmp / "faux.gguf"
    faux_gguf.write_bytes(bytes(range(256)) * 4096)  # 1 Mo
    sortie = tmp / "test.aef"
    forger(faux_gguf, sortie)
    return sortie
