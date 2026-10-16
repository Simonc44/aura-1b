"""Interface terminal de RontoGPT (OpenTUI), branchee DANS le programme.

`rontogpt` sans argument ouvre cette interface. Le cerveau et l'API
(`aura/api.py`) tournent dans CE processus, dans un fil daemon ; le TUI
TypeScript (`tui/`) parle a l'API en HTTP sur 127.0.0.1:8760. Aucune logique
n'est dupliquee : routage, experts, memoire et RAG restent cote Python.

Si Bun ou `tui/` manque (clone frais sans `npm install`), on le dit
clairement et on rend 127 : l'appelant retombe sur la console texte.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_RACINE = Path(__file__).resolve().parent.parent        # racine du projet
_TUI = _RACINE / "tui"
_BUN = _TUI / "node_modules" / "bun" / "bin" / "bun.exe"
_BUN_NIX = _TUI / "node_modules" / ".bin" / "bun"
_INDEX = _TUI / "src" / "index.ts"

PORT = int(os.environ.get("AURA_API_PORT", "8760"))
URL = os.environ.get("AURA_API_URL", f"http://127.0.0.1:{PORT}")

#: rendu par `lancer()` quand l'interface n'est pas installable : l'appelant
#: doit alors retomber sur un autre chemin (console texte), pas planter.
INDISPONIBLE = 127


def _bun_systeme() -> Path | None:
    """Bun deja present dans le PATH (installation systeme), a defaut du local."""
    trouve = shutil.which("bun")
    return Path(trouve) if trouve else None


def chemin_bun() -> Path | None:
    """Le binaire Bun installe localement (`npm install` dans tui/)."""
    if os.name == "nt":
        return _BUN if _BUN.exists() else _bun_systeme()
    return _BUN_NIX if _BUN_NIX.exists() else _bun_systeme()


def raison_indisponible() -> str | None:
    """None si le TUI est lancable ; sinon la raison, en une phrase."""
    if chemin_bun() is None:
        return ("Bun n'est pas installe pour l'interface terminal : ouvre un "
                "terminal dans tui/ et tape « npm install ».")
    if not _INDEX.exists():
        return f"programme du TUI introuvable : {_INDEX}"
    return None


def _api_repond(delai: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/v1/health", timeout=delai) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _build_distant(delai: float = 1.5) -> str | None:
    try:
        with urllib.request.urlopen(f"{URL}/v1/health", timeout=delai) as r:
            return json.loads(r.read().decode()).get("build")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _assurer_api(attente: float = 40.0) -> bool:
    """L'API doit repondre : on la demarre dans CE processus si besoin.

    Une API deja ouverte sur le port mais d'un AUTRE build (ancien processus
    oublie) n'est JAMAIS reutilisee : elle execute du vieux code, donc les
    correctifs n'ont aucun effet. On demarre alors la notre sur un port libre.
    """
    global PORT, URL
    from . import api                                 # import paresseux
    if _api_repond():
        distant = _build_distant()
        if distant == api._BUILD_DEMARRAGE or "AURA_API_URL" in os.environ:
            return True
        PORT = _port_libre()
        URL = f"http://127.0.0.1:{PORT}"
        print("[tui] une API ANCIENNE (build %s, attendu %s) occupait le port : "
              "je demarre une API a jour sur le port %d"
              % (distant or "inconnu", api._BUILD_DEMARRAGE, PORT))
    try:
        api.demarrer_en_tache(port=PORT)
        print(f"[tui] API locale demarree dans ce processus (port {PORT})")
    except OSError as e:
        # Port occupe : un autre RontoGPT tourne (ou une API zombie). Dans
        # les deux cas on ne demarre pas une deuxieme API : on s'y connecte.
        print(f"[tui] port {PORT} deja occupe ({e}) - je m'y connecte")
    fin = time.time() + attente
    while time.time() < fin:
        if _api_repond():
            return True
        time.sleep(0.5)
    return False


def _installer_si_besoin() -> None:
    """Clone frais : `npm install` dans tui/ (pose Bun et OpenTUI en local)."""
    if chemin_bun() is not None or not (_TUI / "package.json").exists():
        return
    npm = shutil.which("npm")
    if npm is None:
        return
    print("[tui] premier lancement : installation de l'interface (npm install)...")
    subprocess.call([npm, "install", "--no-audit", "--no-fund"], cwd=str(_TUI))


def lancer(argv: list[str] | None = None) -> int:
    """Ouvre le TUI. Rend 127 si l'interface n'est pas installable."""
    _installer_si_besoin()
    raison = raison_indisponible()
    if raison is not None:
        print(f"[tui] {raison}", file=sys.stderr)
        return INDISPONIBLE
    if not _assurer_api():
        print(f"[tui] l'API ne repond pas sur {URL} : interface impossible.",
              file=sys.stderr)
        return 1
    bun = chemin_bun()
    if bun is None:                       # course improbable : on reverifie
        return INDISPONIBLE
    # AURA_API_URL transmise au TUI : il vise la meme API que nous, meme si
    # l'utilisateur a change de port via AURA_API_PORT.
    env = {**os.environ, "AURA_API_URL": URL}
    try:
        return subprocess.call([str(bun), str(_INDEX), *(argv or [])],
                               cwd=str(_RACINE), env=env)
    except KeyboardInterrupt:              # Ctrl+C pendant le rendu du TUI
        return 0
