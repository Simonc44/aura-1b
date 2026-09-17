"""Fixtures partagees : mini .aef forge rapide (faux GGUF de 1 Mo)."""
import pytest


@pytest.fixture(scope="session")
def aef_forge(tmp_path_factory):
    from scripts.forger_aef import forger
    tmp = tmp_path_factory.mktemp("aef")
    faux_gguf = tmp / "faux.gguf"
    faux_gguf.write_bytes(bytes(range(256)) * 4096)  # 1 Mo
    sortie = tmp / "test.aef"
    forger(faux_gguf, sortie)
    return sortie
