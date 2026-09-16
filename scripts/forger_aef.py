"""Supercompilateur : forge le fichier monolithique .aef (Aura Executable Format) v2.

Format binaire (little-endian) :

  SECTION HEADER (148 octets, 16 champs)
    magic          4s   b"AURA"
    version        3B   majeur, mineur, correctif
    reserved       B    (alignement)
    build          8s   hash court du commit git
    flags          I    bit 0 : poids compressee (0 = GGUF brut, chargement direct)
    cfg_ctx        I    n_ctx compile
    cfg_batch      I    n_batch compile
    cfg_threads    I    n_threads compile
    taille_config  I    octets de la section config (JSON compresse LZMA)
    taille_code    I    octets de la section code (ZIP du paquet aura + tests)
    taille_gguf    Q    octets du GGUF brut (copie exacte, byte-for-byte)
    sha256_config  32s  integrite config
    sha256_code    32s  integrite code
    sha256_gguf    32s  integrite poids (le boot refuse tout fichier altere)

  SECTION CONFIG  (JSON compresse LZMA : routeur, niveau 0, reglages)
  SECTION CODE    (archive ZIP du paquet `aura/` : le systeme lui-meme)
  SECTION GGUF    (poids BRUTS : llama.cpp les charge directement, zero-copy)

Pourquoi les poids ne sont PAS compresses (mesure sur ce GGUF) :
  LZMA preset 9 ne gagne que 1.6% sur les tenseurs Q4 (haute entropie),
  contre ~7 min de forge et ~2 min de decompression au boot. Le GGUF brut
  permet aussi le chargement direct en RAM sans copie intermediaire.

Usage :
  python scripts/forger_aef.py                    # forge aura_system.aef
  python -m aura.kernel aura_system.aef "question"  # boot + question
"""
import hashlib
import json
import lzma
import os
import struct
import sys
import zipfile
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

MAGIC = b"AURA"
VERSION = (0, 7, 8)
HEADER_FORMAT = "<4sBBBB8sIIIIIQQ32s32s32s"   # 16 champs, 148 octets
HEADER_TAILLE = struct.calcsize(HEADER_FORMAT)

GGUF_DEFAUT = RACINE / "modeles" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
SORTIE = RACINE / "aura_system.aef"


def _sha256_stream(f, taille: int) -> bytes:
    h = hashlib.sha256()
    reste = taille
    while reste > 0:
        bloc = f.read(min(1024 * 1024, reste))
        if not bloc:
            raise IOError("fichier tronque pendant le hachage")
        h.update(bloc)
        reste -= len(bloc)
    return h.digest()


def _pack_code() -> bytes:
    """Archive ZIP du paquet aura/ (le systeme complet, sans les caches)."""
    import io
    buf = io.BytesIO()
    exclusions = {"__pycache__", ".cache_routeur", ".cache_reponses.jsonl"}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for chemin in sorted((RACINE / "aura").rglob("*")):
            if chemin.is_file() and not (set(chemin.parts) & exclusions):
                z.write(chemin, chemin.relative_to(RACINE))
    return buf.getvalue()


def _pack_config() -> bytes:
    """Config compilee : reglages CPU mesurees + reglages du systeme."""
    cfg = {
        "n_ctx": 1536, "n_batch": 768, "n_threads": os.cpu_count() or 4,
        "flash_attn": True, "type_kv": 8,
        "quantification": "Q4_K_M",
        "niveau0_seuil": 0.85,
        "mode_riche_seuil_mots": 9,
        "max_tokens_faits": 120, "max_tokens_explications": 320,
        "build": ".".join(map(str, VERSION)),
    }
    return lzma.compress(json.dumps(cfg).encode("utf-8"), preset=6)


def forger(gguf: Path = GGUF_DEFAUT, sortie: Path = SORTIE) -> Path:
    if not gguf.exists():
        raise FileNotFoundError(
            f"GGUF introuvable : {gguf} (scripts/telecharger_llama.py)")

    print("== AURA FORGE ==")
    config = _pack_config()
    code = _pack_code()
    taille_gguf = gguf.stat().st_size
    print(f"  config : {len(config):,} o | code : {len(code):,} o | "
          f"poids : {taille_gguf:,} o")

    with open(gguf, "rb") as f:
        sha_gguf = _sha256_stream(f, taille_gguf)

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC, VERSION[0], VERSION[1], VERSION[2], 0,
        b"8531f71\x00",
        0,                       # flags : poids brutes (chargement direct)
        1536, 768, os.cpu_count() or 4,
        len(config), len(code), taille_gguf,
        hashlib.sha256(config).digest(),
        hashlib.sha256(code).digest(),
        sha_gguf,
    )

    print("  ecriture monolithique...")
    with open(sortie, "wb") as out, open(gguf, "rb") as src:
        out.write(header)
        out.write(config)
        out.write(code)
        reste = taille_gguf
        while reste > 0:
            bloc = src.read(min(4 * 1024 * 1024, reste))
            if not bloc:
                raise IOError("GGUF tronque pendant la forge")
            out.write(bloc)
            reste -= len(bloc)

    taille_mo = sortie.stat().st_size / (1024 * 1024)
    print(f"== forge OK : {sortie.name} ({taille_mo:.1f} Mo) ==")
    return sortie


if __name__ == "__main__":
    gguf = Path(sys.argv[1]) if len(sys.argv) > 1 else GGUF_DEFAUT
    sys.exit(forger(gguf) and 0)
