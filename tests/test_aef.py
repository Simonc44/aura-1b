"""Tests du format .aef : forge, boot, integrite cryptographique."""
import hashlib
import struct

import pytest

from scripts.forger_aef import HEADER_FORMAT, HEADER_TAILLE, MAGIC, VERSION
from aura import kernel


@pytest.fixture(scope="module")
def aef_forge(tmp_path_factory):
    """Forge un mini .aef (faux GGUF de 1 Mo) pour tester vite."""
    from scripts.forger_aef import forger
    tmp = tmp_path_factory.mktemp("aef")
    faux_gguf = tmp / "faux.gguf"
    faux_gguf.write_bytes(bytes(range(256)) * 4096)  # 1 Mo
    sortie = tmp / "test.aef"
    forger(faux_gguf, sortie)
    return sortie


class TestFormat:
    def test_header_taille_stable(self):
        assert HEADER_TAILLE == 148
        assert struct.calcsize(HEADER_FORMAT) == HEADER_TAILLE

    def test_forge_cree_fichier(self, aef_forge):
        assert aef_forge.exists()
        assert aef_forge.stat().st_size > 1_000_000

    def test_magic_en_tete(self, aef_forge):
        assert aef_forge.read_bytes()[:4] == MAGIC


class TestBoot:
    def test_boot_extrait_tout(self, aef_forge, monkeypatch, tmp_path):
        # ouvrir_et_verifier = la logique crypto + extraction sans re-import
        # du paquet (le re-import de boot() polluerait sys.modules en tests)
        monkeypatch.setattr(kernel, "CACHE", tmp_path / "cache")
        cfg = kernel.ouvrir_et_verifier(aef_forge)
        assert cfg["n_ctx"] == 1536
        assert (tmp_path / "cache" / "cerveau.gguf").exists()
        assert (tmp_path / "cache" / "config.json").exists()
        # 2e appel : le cache est reutilise (pas de re-extraction)
        cfg2 = kernel.ouvrir_et_verifier(aef_forge)
        assert cfg2 == cfg

    def test_altération_poids_refusee(self, aef_forge, monkeypatch, tmp_path):
        monkeypatch.setattr(kernel, "CACHE", tmp_path / "cache2")
        corrompu = aef_forge.parent / "corrompu.aef"
        data = bytearray(aef_forge.read_bytes())
        data[-100] ^= 0xFF            # un seul octet des poids altere
        corrompu.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="poids"):
            kernel.boot(corrompu)

    def test_magic_invalide_refuse(self, tmp_path):
        faux = tmp_path / "faux.aef"
        faux.write_bytes(b"PASD" + b"\x00" * 200)
        with pytest.raises(ValueError, match="magic"):
            kernel.boot(faux)
