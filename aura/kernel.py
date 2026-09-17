"""Aura Kernel : boot un fichier .aef (Aura Executable Format), chiffre ou non.

Sequence de demarrage (ouvrir_et_verifier) :
1. lecture du magic : "AURA" (clair) ou "AUAE" (chiffre AES-256-GCM)
2. si chiffre : dechiffrement en memoire (PBKDF2 600k iterations) — le secret
   n'est JAMAIS stocke dans le fichier
3. verification des 3 checksums SHA-256 (config, code, poids) :
   un seul octet altere = refus de boot (integrite cryptographique)
4. extraction de la config (JSON), du code (paquet aura) et des poids (GGUF)
   dans un cache local (.cache_aef/), seulement si le hash a change

Puis, selon le contexte :
- boot()            : import du paquet extrait + config compilee (usage python)
- boot_depuis_cache(): import direct depuis un cache deja rempli (mode exe :
  Aura.exe verifie/dechiffre/extrait, puis delegue a l'environment python
  local qui possede les dependances lourdes : numpy, llama-cpp, ...)

Usage :
  python -m aura.kernel aura_system.aef "question"
  python -m aura.kernel aura_system.aef              # mode interactif
  python -m aura.kernel --cache "question"           # cache deja rempli (exe)
  Aura.exe aura_system.aef "question"                # exe (delegue a uv)
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

# En exe (gele), __file__ pointe vers l'extraction temporaire : la racine du
# projet est le dossier qui contient l'exe (et le .aef a cote).
if getattr(sys, "frozen", False):
    RACINE = Path(sys.executable).resolve().parent
else:
    RACINE = Path(__file__).resolve().parent.parent
# AURA_CACHE : l'exe leger le transmet au python delegue -> meme cache
cache_env = os.environ.get("AURA_CACHE")
CACHE = Path(cache_env) if cache_env else RACINE / ".cache_aef"


def _importer_chiffrement():
    """Import du module chiffrement : paquet (python) ou top-level (exe)."""
    try:
        from . import chiffrement
        return chiffrement
    except ImportError:
        import chiffrement
        return chiffrement


def _ouvrir(path_aef: Path):
    """Renvoie un flux binaire du .aef CLAIR.

    - .aef clair    : le fichier, ouvert directement (zero copie)
    - .aef.chiffre  : dechiffre ENTIEREMENT en memoire (AES-256-GCM), le flux
                      est un BytesIO ; le secret n'est jamais stocke sur disque
    """
    ch = _importer_chiffrement()
    f = open(path_aef, "rb")
    try:
        magic = f.read(4)
    except Exception:
        f.close()
        raise
    if magic == ch.MAGIC_CHIFFRE:
        f.close()
        secret = os.environ.get("AURA_SECRET")
        if not secret:
            import getpass
            secret = getpass.getpass("Fichier chiffre. Secret : ")
        print("[kernel] dechiffrement AES-256-GCM (PBKDF2 600k itérations)...")
        t0 = time.time()
        clair = ch.dechiffrer(path_aef.read_bytes(), secret)
        print(f"[kernel] dechiffre en {time.time() - t0:.1f}s, integrite AEAD OK")
        return io.BytesIO(clair)
    if magic != b"AURA":
        f.close()
        raise ValueError("magic invalide : pas un fichier Aura (.aef ou .aef.chiffre)")
    f.seek(0)
    return f


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


def ouvrir_et_verifier(path_aef: str | Path) -> dict:
    """Verifie, dechiffre si besoin, extrait en cache. Renvoie la config.

    N'importe AUCUN module lourd : utilisable depuis l'exe leger.
    """
    path_aef = Path(path_aef)
    f = _ouvrir(path_aef)
    try:
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

        # 1. config (LZMA) : verifie puis decompresse
        cfg_comp = f.read(t_cfg)
        if hashlib.sha256(cfg_comp).digest() != sha_cfg:
            raise ValueError("config alteree : refus de boot")
        cfg = json.loads(lzma.decompress(cfg_comp))

        # 2. code (ZIP) : verifie puis extrait si nouveau
        code = f.read(t_code)
        if hashlib.sha256(code).digest() != sha_code:
            raise ValueError("code altere : refus de boot")

        # 3. poids (GGUF brut) : integrite en flux, SANS decompression
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
    finally:
        f.close()

    # 4. extraction du code si le hash a change
    dossier_code = CACHE / "code"
    marqueur = CACHE / "version"
    if not marqueur.exists() or marqueur.read_text() != sha_code.hex() \
            or not (dossier_code / "aura").exists():
        dossier_code.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(code)) as z:
            z.extractall(dossier_code)
        marqueur.write_text(sha_code.hex())

    CACHE.mkdir(exist_ok=True)
    (CACHE / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


def boot(path_aef: str | Path) -> dict:
    """Verification complete + import du systeme (usage python)."""
    cfg = ouvrir_et_verifier(path_aef)
    dossier_code = CACHE / "code"
    sys.path.insert(0, str(dossier_code))
    # le CODE EXTRAIT du .aef prime sur toute copie deja importee (exe ou
    # paquet source) : c'est le systeme signe dans le fichier qui s'execute
    for nom in [m for m in list(sys.modules)
                if m == "aura" or m.startswith("aura.")]:
        del sys.modules[nom]
    aura = importlib.import_module("aura")
    os.environ["AURA_CONFIG"] = json.dumps(cfg)   # le cerveau lit ce chemin
    print(f"[kernel] config compilee : ctx={cfg['n_ctx']} batch={cfg['n_batch']} "
          f"threads={cfg['n_threads']} flash_attn={cfg['flash_attn']}")
    return {"module": aura, "config": cfg}


def boot_depuis_cache() -> dict:
    """Import direct depuis un cache deja rempli par l'exe (mode delegue)."""
    fichier_cfg = CACHE / "config.json"
    if not fichier_cfg.exists() or not (CACHE / "code" / "aura").exists():
        raise ValueError("cache vide : boote d'abord le .aef "
                         "(Aura.exe <fichier.aef> ou python -m aura.kernel <fichier.aef>)")
    cfg = json.loads(fichier_cfg.read_text(encoding="utf-8"))
    sys.path.insert(0, str(CACHE / "code"))
    aura = importlib.import_module("aura")
    os.environ["AURA_CONFIG"] = json.dumps(cfg)
    print(f"[kernel] boot depuis le cache : ctx={cfg['n_ctx']} "
          f"batch={cfg['n_batch']} threads={cfg['n_threads']}")
    return {"module": aura, "config": cfg}


def principal() -> int:
    if len(sys.argv) < 2:
        print("usage : python -m aura.kernel <fichier.aef> [question]\n"
              "        python -m aura.kernel --cache [question]   (cache deja rempli)")
        return 2

    if sys.argv[1] == "--cache":
        sysm = boot_depuis_cache()
        questions = sys.argv[2:]
    else:
        sysm = boot(sys.argv[1])
        questions = sys.argv[2:]

    ia = sysm["module"].Aura1B()
    if questions:
        print(ia.executer(" ".join(questions)))
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
