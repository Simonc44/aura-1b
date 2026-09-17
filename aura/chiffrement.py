"""Chiffrement AES-256-GCM du fichier .aef — publication confidentielle.

Objectif : publier aura_system.aef sur Hugging Face SANS exposer le code
ni permettre une copie modifiable.

Concept :
  - la forge produit le .aef clair (usage local) puis, option --chiffrer,
    l'emballe dans un conteneur AES-256-GCM (AEAD : confidentialite +
    integrite : un octet modifie = echec de dechiffrement, pas de boot).
  - la cle derive du secret de l'utilisateur (PBKDF2-HMAC-SHA256,
    600 000 iterations, sel aleatoire 16 o) : jamais stockee dans le fichier.
  - le kernel ouvre les deux formats indifferemment (magic "AURA" clair,
    magic "AUAE" chiffre).

Format .aef chiffre :
  magic "AUAE" | 1 o version | 16 o sel | 12 o nonce | 8 o taille claire
  | payload AES-GCM (le .aef complet, header compris)

La verification SHA-256 du .aef interne reste active APRES dechiffrement :
deux couches d'integrite independantes.
"""
import hashlib
import os
import struct

MAGIC_CHIFFRE = b"AUAE"
VERSION = 1
SEL_TAILLE = 16
NONCE_TAILLE = 12
_PBKDF2_ITERATIONS = 600_000


def deriver_cle(secret: str, sel: bytes) -> bytes:
    """Cle 32 octets depuis le secret (PBKDF2-HMAC-SHA256)."""
    return hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"), sel,
                               _PBKDF2_ITERATIONS, dklen=32)


def chiffrer(donnees: bytes, secret: str) -> bytes:
    """Chiffre un .aef complet -> conteneur AUAE (AES-256-GCM)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    sel = os.urandom(SEL_TAILLE)
    nonce = os.urandom(NONCE_TAILLE)
    cle = deriver_cle(secret, sel)
    # AAD = en-tete : lie le payload a SA metadonnee (anti permutation)
    aad = MAGIC_CHIFFRE + struct.pack("<BQ", VERSION, len(donnees))
    payload = AESGCM(cle).encrypt(nonce, donnees, aad)
    return (MAGIC_CHIFFRE + struct.pack("<B", VERSION) + sel + nonce
            + struct.pack("<Q", len(donnees)) + payload)


def dechiffrer(conteneur: bytes, secret: str) -> bytes:
    """Dechiffre un conteneur AUAE -> le .aef clair. Echec = fichier altere
    ou mauvais mot de passe (impossible de distinguer, c'est le but)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    entete = MAGIC_CHIFFRE + struct.pack("<B", VERSION)
    if conteneur[:len(entete)] != entete:
        raise ValueError("pas un conteneur AUAE")
    off = len(entete)
    sel = conteneur[off:off + SEL_TAILLE];            off += SEL_TAILLE
    nonce = conteneur[off:off + NONCE_TAILLE];        off += NONCE_TAILLE
    (taille_claire,) = struct.unpack("<Q", conteneur[off:off + 8]); off += 8
    payload = conteneur[off:]
    cle = deriver_cle(secret, sel)
    aad = MAGIC_CHIFFRE + struct.pack("<BQ", VERSION, taille_claire)
    try:
        clair = AESGCM(cle).decrypt(nonce, payload, aad)
    except Exception as e:
        raise ValueError("dechiffrement impossible : mot de passe errone "
                         "ou fichier altere") from e
    if len(clair) != taille_claire:
        raise ValueError("taille incoherente : fichier altere")
    return clair


def est_chiffre(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == MAGIC_CHIFFRE
