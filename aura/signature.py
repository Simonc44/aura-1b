"""Signature numerique Ed25519 des fichiers .aef — provenance authentifiee.

Ce que la signature apporte (au-dela du SHA-256 interne) :
  - le SHA-256 prouve que le fichier n'a PAS ete modifie ;
  - la signature prouve QUI l'a forge : seule la cle privee de l'auteur
    peut produire une signature valide. Copier le fichier ne copie pas
    l'identite : un tiers ne peut PAS forger un .aef qui passe la
    verification avec notre cle publique embarquee.

Format du fichier de signature  (<fichier>.sig) :
  magic        4s   b"AUSG"
  version      B    1
  t_id         B    longueur de l'identite du signataire
  identite     t_id identite en utf-8 (ex. "Simonc44")
  cle_publique 32   Ed25519 brute (RFC 8032)
  signature    64   Ed25519 sur les 32 octets du SHA-256 du payload

Le kernel embarque un CACHE de cles de confiance (cle_publique -> identite).
Un .aef non signe boote normalement (la signature est une couche de plus,
pas une contrainte) ; un .aef signe par une cle inconnue est REFUSE —
politique de confiance stricte.

La cle privee (fichier PEM PKCS#8) ne quitte JAMAIS la machine de forge.
"""
import base64
import hashlib
import os
import struct
from pathlib import Path

MAGIC_SIGNATURE = b"AUSG"
VERSION_SIGNATURE = 1

# Cle de confiance : la cle publique d'Aura (Simonc44). Toute signature
# hors de cette cle est REFUSEE au boot. Pour la remplacer, colle la cle
# publique (hex) dans AURA_TRUST_KEY ou ici.
_CLE_CONFIANCE_DEFAUT = "b18db698206a32b05c743f71f33994af664227d948863a99b85e8b93fc03032e"

# Import tolerant : cryptography est une dependance de la forge/kernel,
# jamais du paquet installe chez l'utilisateur final.
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    _DISPO = True
except ImportError:                      # pragma: no cover
    Ed25519PrivateKey = Ed25519PublicKey = None
    _DISPO = False


def disponible() -> bool:
    return _DISPO


# ------------------------------------------------------------------
# Gestion de la paire de cles
# ------------------------------------------------------------------

def generer_paire(chemin_privee: str | Path = "cle_privee.pem",
                  chemin_publique: str | Path = "cle_publique.pub",
                  ) -> tuple[Path, Path]:
    """Genere une paire Ed25519. Privee en PKCS#8 (PEM, sans mot de passe :
    protege le fichier par les permissions du disque, pas par un secret)."""
    if not _DISPO:
        raise RuntimeError("cryptography indisponible : pip install cryptography")
    cle = Ed25519PrivateKey.generate()
    p_privee = Path(chemin_privee)
    p_publique = Path(chemin_publique)

    pem_privee = cle.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub = cle.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    p_privee.write_bytes(pem_privee)
    p_publique.write_text(base64.b64encode(pub).decode("ascii"))
    return p_privee, p_publique


def charger_cle_privee(chemin: str | Path) -> "Ed25519PrivateKey":
    if not _DISPO:
        raise RuntimeError("cryptography indisponible")
    from cryptography.hazmat.primitives import serialization
    return serialization.load_pem_private_key(
        Path(chemin).read_bytes(), password=None)


def cle_privee_vers_publique_brute(cle_privee) -> bytes:
    return cle_privee.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


# ------------------------------------------------------------------
# Signer / verifier
# ------------------------------------------------------------------

def signer(sha256_payload: bytes, cle_privee, identite: str = "Simonc44") -> bytes:
    """Bloc de signature complet (le contenu du fichier .sig).

    sha256_payload : les 32 octets du SHA-256 du fichier .aef (clair).
    """
    if not _DISPO:
        raise RuntimeError("cryptography indisponible")
    id_b = identite.encode("utf-8")
    if len(id_b) > 255:
        raise ValueError("identite trop longue (255 max)")
    cle_pub = cle_privee_vers_publique_brute(cle_privee)
    sig = cle_privee.sign(sha256_payload)
    return (MAGIC_SIGNATURE
            + struct.pack("<BB", VERSION_SIGNATURE, len(id_b))
            + id_b + cle_pub + sig)


def ecrire_signature(path_aef: str | Path, cle_privee, identite: str = "Simonc44") -> Path:
    """Signe le .aef et ecrit <path>.sig. Renvoie le chemin du .sig."""
    path_aef = Path(path_aef)
    h = _sha256_fichier(path_aef)
    bloc = signer(h, cle_privee, identite)
    path_sig = path_aef.with_suffix(".aef.sig")
    path_sig.write_bytes(bloc)
    return path_sig


def lire_bytes(brut: bytes) -> dict:
    """Parse un bloc de signature depuis des octets.
    Renvoie {identite, cle_publique, signature}."""
    if brut[:4] != MAGIC_SIGNATURE or len(brut) < 6:
        raise ValueError("pas un bloc de signature Aura (.sig)")
    version, t_id = struct.unpack("<BB", brut[4:6])
    off = 6
    identite = brut[off:off + t_id].decode("utf-8", "replace"); off += t_id
    cle_pub = brut[off:off + 32];                      off += 32
    signature_o = brut[off:off + 64];                  off += 64
    if off != len(brut):
        raise ValueError("bloc .sig tronque ou invalide")
    return {"identite": identite, "cle_publique": cle_pub,
            "signature": signature_o}


def lire_signature(path_sig: str | Path) -> dict:
    """Parse un bloc de signature depuis un fichier .sig."""
    return lire_bytes(Path(path_sig).read_bytes())


def verifier_signature(sha256_payload: bytes, bloc: dict) -> bool:
    """Vrai si la signature correspond au hash ET a la cle du bloc."""
    if not _DISPO:
        return False
    from cryptography.exceptions import InvalidSignature
    cle = Ed25519PublicKey.from_public_bytes(bloc["cle_publique"])
    try:
        cle.verify(bloc["signature"], sha256_payload)
        return True
    except InvalidSignature:
        return False


# ------------------------------------------------------------------
# Politique de confiance du kernel
# ------------------------------------------------------------------

def cle_de_confiance() -> bytes | None:
    """La cle publique de confiance (hex via AURA_TRUST_KEY ou defaut code)."""
    hexa = os.environ.get("AURA_TRUST_KEY") or _CLE_CONFIANCE_DEFAUT
    if not hexa:
        return None
    try:
        return bytes.fromhex(hexa)
    except ValueError:
        return None


def verifier_confiance(bloc: dict, cle_attendue: bytes | None = None) -> bool:
    """La signature provient-elle d'une cle de confiance ?

    Politique du kernel : un .aef SIGNE par une cle inconnue = refuse
    (usurpation d'identite possible). Non signe = OK (pas de signature).
    """
    if cle_attendue is None:
        cle_attendue = cle_de_confiance()
    if cle_attendue is None:
        # Pas de cle de confiance configuree : on accepte toute signature
        # valide du point de vue crypto (le hash a deja fait l'integrite).
        return True
    return bloc["cle_publique"] == cle_attendue


# ------------------------------------------------------------------

def _sha256_fichier(path: Path) -> bytes:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloc in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloc)
    return h.digest()


def empreinte_publique(cle_pub_brute: bytes) -> str:
    """Empreinte lisible (8 hex) pour affichage."""
    return hashlib.sha256(cle_pub_brute).hexdigest()[:8]

