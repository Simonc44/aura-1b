"""Fixtures partagees : mini .aef forge rapide (faux GGUF de 1 Mo)."""
import pytest


@pytest.fixture(autouse=True)
def _apis_publiques_coupees(monkeypatch):
    """Aucun reseau : les API publiques sont coupees dans TOUS les tests.

    Les tests dedieses de aura/apis_publiques.py re-active la brique avec
    monkeypatch.setenv("AURA_APIS", "1") + un urllib mocke.
    """
    monkeypatch.setenv("AURA_APIS", "0")


@pytest.fixture(scope="session")
def aef_forge(tmp_path_factory):
    from scripts.forger_aef import forger
    tmp = tmp_path_factory.mktemp("aef")
    faux_gguf = tmp / "faux.gguf"
    faux_gguf.write_bytes(bytes(range(256)) * 4096)  # 1 Mo
    sortie = tmp / "test.aef"
    forger(faux_gguf, sortie)
    return sortie
