"""Aura Kernel : boot un fichier .aef (Aura Executable Format).

Sequence de demarrage :
1. lecture du header (72 octets), verification du magic + version
2. verification des 3 checksums SHA-256 (config, code, poids) :
   un seul octet altere = refus de boot (integrite cryptographique)
3. extraction de la config (reglages CPU compilees) et du code (paquet aura)
   dans un cache local (.cache_aef/), seulement si le hash a change
4. import du paquet extrait et boot d'Aura1B avec la config compilee

Usage :
  python -m aura.kernel aura_system.aef "question"
  python -m aura.kernel aura_system.aef            # mode interactif
"""
import hashlib
import importlib
import io
import json
import lzma
import os
import struct
import sys
import time
import zipfile
from pathlib import Path

HEADER_FORMAT = "<4sBBBB8sIIIIIQQ32s32s32s"
HEADER_TAILLE = struct.calcsize(HEADER_FORMAT)

RACINE = Path(__file__).resolve().parent.parent
CACHE = RACINE / ".cache_aef"


def _sha256_stream(f, taille: int) -> bytes:
    h = hashlib.sha256()
    reste = taille
    while reste > 0:
        bloc = f.read(min(1024 * 1024, reste))
        if not bloc:
            raise IOError("section tronquee (fichier altere ?)")
        h.update(bloc)
        reste -= len(bloc)
    return h.digest()


def boot(path_aef: str | Path) -> dict:
    """Verifie, extrait et boote le systeme. Renvoie (module aura, config)."""
    path_aef = Path(path_aef)
    with open(path_aef, "rb") as f:
        header = f.read(HEADER_TAILLE)
        if len(header) != HEADER_TAILLE:
            raise ValueError("fichier trop court : pas un .aef")

        (magic, majeur, mineur, correctif, _res, build, flags,
         cfg_ctx, cfg_batch, cfg_threads,
         t_cfg, t_code, t_gguf,
         sha_cfg, sha_code, sha_gguf) = struct.unpack(HEADER_FORMAT, header)

        if magic != b"AURA":
            raise ValueError("magic invalide : pas un fichier Aura (.aef)")
        build_txt = build.rstrip(b"\x00").decode("ascii", "replace")
        print(f"[kernel] Aura v{majeur}.{mineur}.{correctif} (build {build_txt})")

        # 1. config (LZMA) : petit, verifie puis decompresse
        cfg_comp = f.read(t_cfg)
        if hashlib.sha256(cfg_comp).digest() != sha_cfg:
            raise ValueError("config alteree : refus de boot")
        cfg = json.loads(lzma.decompress(cfg_comp))

        # 2. code (ZIP) : verifie puis extrait si nouveau
        code = f.read(t_code)
        if hashlib.sha256(code).digest() != sha_code:
            raise ValueError("code altere : refus de boot")

        # 3. poids (GGUF brut) : integrite stream, SANS decompression
        print(f"[kernel] verification des poids ({t_gguf:,} octets)...")
        sha_calc = _sha256_stream(f, t_gguf)
        if sha_calc != sha_gguf:
            raise ValueError("poids alters : refus de boot (integrite rompue)")

        cache_gguf = CACHE / "cerveau.gguf"
        if cache_gguf.exists() and cache_gguf.stat().st_size == t_gguf:
            print("[kernel] poids deja en cache, copie ignoree")
        else:
            CACHE.mkdir(exist_ok=True)
            print("[kernel] extraction des poids (1re fois seulement)...")
            f.seek(HEADER_TAILLE + t_cfg + t_code)   # rembobine : le hash a
            # deja consomme la section poids, il faut la relire depuis le debut
            with open(cache_gguf, "wb") as out:
                reste = t_gguf
                while reste > 0:
                    bloc = f.read(min(4 * 1024 * 1024, reste))
                    if not bloc:
                        raise IOError("section poids tronquee : fichier altere")
                    out.write(bloc)
                    reste -= len(bloc)

    # 4. extraction du code si le hash a change
    dossier_code = CACHE / "code"
    marqueur = CACHE / "version"
    if not marqueur.exists() or marqueur.read_text() != sha_code.hex() \
            or not (dossier_code / "aura").exists():
        dossier_code.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(code)) as z:
            z.extractall(dossier_code)
        marqueur.write_text(sha_code.hex())

    # 5. import du paquet extrait (le systeme lui-meme) + config compilee
    dossier_cache = CACHE / "paquet"
    dossier_cache.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(dossier_code))
    aura = importlib.import_module("aura")

    os.environ["AURA_CONFIG"] = json.dumps(cfg)   # le cerveau lit ce chemin
    print(f"[kernel] config compilee : ctx={cfg['n_ctx']} batch={cfg['n_batch']} "
          f"threads={cfg['n_threads']} flash_attn={cfg['flash_attn']}")
    return {"module": aura, "config": cfg}


def principal() -> int:
    if len(sys.argv) < 2:
        print("usage : python -m aura.kernel <fichier.aef> [question]")
        return 2
    sysm = boot(sys.argv[1])
    ia = sysm["module"].Aura1B()
    if len(sys.argv) > 2:
        print(ia.executer(" ".join(sys.argv[2:])))
        return 0
    print('Kernel online. "exit" pour quitter.')
    while True:
        try:
            q = input("toi> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if q.lower() in ("exit", "quit"):
            return 0
        t0 = time.time()
        print(ia.executer(q))
        print(f"[{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    sys.exit(principal())
