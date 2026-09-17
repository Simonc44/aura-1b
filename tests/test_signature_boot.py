"""Tests signature Ed25519 et boot rapide (marqueur de verification)."""
import os
import time

import pytest

from aura import kernel, signature


@pytest.fixture(scope="module")
def paire(tmp_path_factory):
    """Une paire Ed25519 pour tout le module."""
    tmp = tmp_path_factory.mktemp("cles")
    privee, publique = signature.generer_paire(tmp / "k.pem", tmp / "k.pub")
    return privee, publique


class TestCles:
    def test_generation_paire(self, paire):
        privee, publique = paire
        assert privee.exists() and publique.exists()
        pub = privee.read_bytes()
        assert b"BEGIN PRIVATE KEY" in pub or b"BEGIN" in pub

    def test_publique_derivee(self, paire):
        privee, _ = paire
        pub_brute = signature.cle_privee_vers_publique_brute(
            signature.charger_cle_privee(privee))
        assert len(pub_brute) == 32


class TestSignature:
    def test_roundtrip_signer_verifier(self, paire):
        privee, _ = paire
        sha = os.urandom(32)
        bloc = signature.signer(sha, signature.charger_cle_privee(privee),
                                identite="Test")
        parse = signature.lire_bytes(bloc)
        assert parse["identite"] == "Test"
        assert signature.verifier_signature(sha, parse)

    def test_mauvais_hash_refuse(self, paire):
        privee, _ = paire
        sha = os.urandom(32)
        bloc = signature.signer(sha, signature.charger_cle_privee(privee))
        parse = signature.lire_bytes(bloc)
        assert not signature.verifier_signature(os.urandom(32), parse)

    def test_autre_cle_refusee(self, paire, tmp_path):
        privee, _ = paire
        autre = signature.generer_paire(tmp_path / "a.pem", tmp_path / "a.pub")
        sha = os.urandom(32)
        bloc = signature.signer(sha, signature.charger_cle_privee(privee))
        parse = signature.lire_bytes(bloc)
        # signe par privee, verifie contre la cle d'une autre paire
        autre_pub = signature.cle_privee_vers_publique_brute(
            signature.charger_cle_privee(autre[0]))
        parse_faux = {**parse, "cle_publique": autre_pub}
        assert not signature.verifier_signature(sha, parse_faux)

    def test_fichier_sig_ecrit_et_verifie(self, paire, tmp_path):
        privee, _ = paire
        cible = tmp_path / "fichier.aef"
        cible.write_bytes(b"payload" * 1000)
        path_sig = signature.ecrire_signature(
            cible, signature.charger_cle_privee(privee), identite="Test")
        assert path_sig.exists()
        sha = signature._sha256_fichier(cible)
        assert signature.verifier_signature(
            sha, signature.lire_signature(path_sig))

    def test_politique_confiance(self, paire):
        privee, _ = paire
        sha = os.urandom(32)
        bloc = signature.signer(sha, signature.charger_cle_privee(privee))
        parse = signature.lire_bytes(bloc)
        # cle de confiance = notre cle : accepte
        notre_cle = signature.cle_privee_vers_publique_brute(
            signature.charger_cle_privee(privee))
        assert signature.verifier_confiance(parse, cle_attendue=notre_cle)
        # cle de confiance = une autre : refuse
        assert not signature.verifier_confiance(
            parse, cle_attendue=os.urandom(32))


class TestBootRapide:
    def test_premier_boot_scelle_marqueur(self, aef_forge, monkeypatch, tmp_path):
        monkeypatch.setattr(kernel, "CACHE", tmp_path / "c1")
        kernel.ouvrir_et_verifier(aef_forge)
        assert (tmp_path / "c1" / "verifie.json").exists()

    def test_boot_rapide_saute_le_hash(self, aef_forge, monkeypatch, tmp_path):
        monkeypatch.setattr(kernel, "CACHE", tmp_path / "c2")
        kernel.ouvrir_et_verifier(aef_forge)          # 1er boot : complet
        # on RALENTIT volontairement le hachage : si le fast path est pris,
        # le re-hash n'a pas lieu et le test rend la main rapidement
        orig = kernel._sha256_stream

        def lent(f, taille):
            time.sleep(2)
            return orig(f, taille)

        monkeypatch.setattr(kernel, "_sha256_stream", lent)
        t0 = time.time()
        kernel.ouvrir_et_verifier(aef_forge)          # 2e boot : fast path
        assert time.time() - t0 < 1.0, "le fast path doit sauter le re-hash"

    def test_modification_detectee(self, aef_forge, monkeypatch, tmp_path):
        monkeypatch.setattr(kernel, "CACHE", tmp_path / "c3")
        kernel.ouvrir_et_verifier(aef_forge)
        # on touche le fichier (mtime change) -> marqueur invalide -> re-hash
        st = aef_forge.stat()
        os.utime(aef_forge, (st.st_atime, st.st_mtime + 10))
        assert not kernel._marqueur_valide(aef_forge, None)
        kernel.ouvrir_et_verifier(aef_forge)          # re-verifie, re-scelle

    def test_verifier_tout_force_le_hash(self, aef_forge, monkeypatch, tmp_path):
        monkeypatch.setattr(kernel, "CACHE", tmp_path / "c4")
        kernel.ouvrir_et_verifier(aef_forge)
        appels = []
        orig = kernel._sha256_stream

        def espion(f, taille, aussi=None):
            appels.append(taille)
            return orig(f, taille, aussi=aussi)

        monkeypatch.setattr(kernel, "_sha256_stream", espion)
        kernel.ouvrir_et_verifier(aef_forge, verifier_tout=True)
        assert appels, "--verifier doit imposer le re-hash"

    def test_invalide_sans_marqueur(self, tmp_path, monkeypatch):
        monkeypatch.setattr(kernel, "CACHE", tmp_path / "vide")
        assert not kernel._marqueur_valide(tmp_path / "nimporte.aef", None)


class TestSignatureIntegree:
    def test_sig_sur_fichier_entier_boot_ok(self, aef_forge, monkeypatch, tmp_path):
        """Le .sig porte sur le hash du FICHIER ENTIER (comme la forge) :
        le boot doit l'accepter (hash cumule en une passe de lecture).
        La cle de confiance embarquee (Simonc44) est neutralisee pour que
        la cle de test soit acceptee."""
        monkeypatch.setattr(signature, "_CLE_CONFIANCE_DEFAUT", "")
        monkeypatch.setattr(kernel, "CACHE", tmp_path / "cs1")
        cle = signature.charger_cle_privee(paire_cle(tmp_path))
        sha_fichier = signature._sha256_fichier(aef_forge)
        bloc = signature.signer(sha_fichier, cle, identite="Test")
        sig = aef_forge.with_suffix(".aef.sig")
        sig.write_bytes(bloc)
        cfg = kernel.ouvrir_et_verifier(aef_forge)      # ne doit PAS lever
        assert cfg["n_ctx"] == 1536
        sig.unlink()

    def test_sig_falsifie_refuse(self, aef_forge, monkeypatch, tmp_path):
        monkeypatch.setattr(signature, "_CLE_CONFIANCE_DEFAUT", "")
        monkeypatch.setattr(kernel, "CACHE", tmp_path / "cs2")
        cle = signature.charger_cle_privee(paire_cle(tmp_path))
        sha_fichier = signature._sha256_fichier(aef_forge)
        bloc = bytearray(signature.signer(sha_fichier, cle))
        bloc[-1] ^= 0xFF                                 # signature falsifiee
        sig = aef_forge.with_suffix(".aef.sig")
        sig.write_bytes(bytes(bloc))
        with pytest.raises(ValueError, match="signature"):
            kernel.ouvrir_et_verifier(aef_forge)
        sig.unlink()


def paire_cle(tmp_path):
    """Charge (ou genere une fois) la cle de test dans tmp_path/k.pem."""
    p = tmp_path / "k.pem"
    if not p.exists():
        signature.generer_paire(p, tmp_path / "k.pub")
    return p
