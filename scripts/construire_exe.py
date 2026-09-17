"""Construit Aura.exe : le kernel .aef en executable Windows leger.

Strategie (pourquoi leger) :
  L'exe n'embarque QUE le kernel (verification SHA-256, dechiffrement
  AES-256-GCM, extraction du .aef) — pas numpy/scipy/llama-cpp. Apres
  verification, il delegue l'execution au python local (uv) qui possede
  les dependances lourdes. Resultat :
    - taille : ~12-15 Mo (runtime CPython + cryptography, sans les libs IA)
    - boot : verification des 807 Mo = quelques secondes, une seule fois
      (ensuite --cache demarre instantanement)

PROTECTION du code (honnete) :
  - l'exe contient du BYTECODE compile (pas de sources lisibles)
  - le .aef publie est CHIFFRE AES-256-GCM : sans le secret, les poids et
    le code compresses sont indistinguables d'un bruit aleatoire
  - deux couches d'integrite : AEAD (dechiffrement) + SHA-256 (header)
  La copie passive est bloquee ; aucun binaire n'est inviolable a 100 %
  (ingenierie inverse toujours possible en theorie) — voir README.

Usage :
  uv run python scripts/construire_exe.py          # -> dist/Aura.exe
  dist\\Aura.exe aura_system.aef.enc "carre de 15"  # verifie+extrait+delegue
  dist\\Aura.exe --cache "carre de 15"              # cache deja rempli
"""
import shutil
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "dist"
NOM = "Aura"

REQUIRED = ["cryptography"]


def _importable(nom: str) -> bool:
    try:
        __import__(nom)
        return True
    except ImportError:
        return False


def verifier_prerequis() -> None:
    manquant = [m for m in REQUIRED if not _importable(m)]
    if manquant:
        raise SystemExit(f"dependances manquantes : {', '.join(manquant)} "
                         f"(uv add {' '.join(manquant)})")


def construire() -> Path:
    verifier_prerequis()

    # kernel.py et chiffrement.py sont importes DYNAMIQUEMENT par winmain
    # (PyInstaller ne les voit pas) -> embarques en top-level, bytecode seul
    # (les .py sources sont exclus), et le paquet aura/ est EXCLUS pour ne
    # jamais entrer en conflit avec le paquet aura extrait du .aef.
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--console",
        "--name", NOM,
        "--clean", "--noconfirm",
        # --add-data : src:dst ; sur Windows le separateur est ';' ; les
        # fichiers .py sont automatiquement EXCLUS de l'archive finale par
        # PyInstaller (seul le bytecode .pyc compile est embarque)
        f"--add-data={RACINE / 'aura' / 'kernel.py'};.",
        f"--add-data={RACINE / 'aura' / 'chiffrement.py'};.",
        "--exclude-module", "aura",
        str(RACINE / "aura" / "winmain.py"),
    ]

    print("== PyInstaller : build leger ==")
    subprocess.run(cmd, cwd=RACINE, check=True)

    exe = SORTIE / f"{NOM}.exe"
    if not exe.exists():
        raise FileNotFoundError(f"exe non produit : {exe}")
    print(f"== OK : {exe} ({exe.stat().st_size / (1024*1024):.1f} Mo) ==")

    # PyInstaller laisse un dossier build/ a nettoyer
    dossier_build = RACINE / "build"
    if dossier_build.exists():
        shutil.rmtree(dossier_build, ignore_errors=True)

    print("""
usage de l'exe :
  Aura.exe <fichier.aef|fichier.aef.enc> [question]   verifie + extrait + repond
  Aura.exe --cache [question]                          cache deja rempli
""")
    return exe


if __name__ == "__main__":
    construire()
