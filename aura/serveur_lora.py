"""Cerveau via llama-server officiel : le chemin LoRA qui FONCTIONNE.

Le hot-swap in-process (adaptateurs.py via ctypes) est casse par un bug
llama.cpp (LoRA x CPU_REPACK, access violation au decode). Le serveur
officiel llama-server (ggml-org) charge plusieurs adaptateurs au boot
et les active/desactive a la volee via POST /lora-adapters (echelles) :
c'est le mecanisme utilise par les pipelines de production.

Protocol (verifie sur b11111) :
- GET  /health          -> {"status":"ok"}
- GET  /lora-adapters   -> [{"id":0,"path":...,"scale":1.0}, ...]
- POST /lora-adapters   -> corps JSON [{"id":N,"scale":0.0|1.0}] (partiel OK)
- POST /v1/chat/completions (OpenAI-compatible)

Cycle de vie (AURA_SERVEUR=1 pour activer) :
- le serveur est DEMARRE par Aura au premier besoin (port AURA_SERVEUR_PORT,
  defaut 8757), binaire serveur/llama-server.exe, base modeles/<cerveau>,
  les 4 adaptateurs serveur/aura-*-T.gguf montes au boot ;
- si un serveur tourne deja sur le port, il est REUTILISE (pas de double
  boot) ;
- echec de boot ou d'appel -> None (l'appelant retombe sur le chemin
  in-process), jamais bloquant.
"""
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import urllib.request

LOG = logging.getLogger("aura.serveur_lora")

_RACINE = Path(__file__).resolve().parent.parent

# categorie routee -> id de l'adaptateur au boot du serveur
_IDS = {"math": 0, "web": 1, "code": 2, "general": 3}

_ADRESSE = os.environ.get("AURA_SERVEUR_ADRESSE", "127.0.0.1:8757")

_process: subprocess.Popen | None = None
_actif: str | None = None          # categorie actuellement a scale=1.0
_indispo: str | None = None        # raison d'une indisponibilite (log 1 fois)


def _actifs() -> bool:
    """Interrupteur : AURA_SERVEUR=0 (defaut) -> chemin in-process."""
    return os.environ.get("AURA_SERVEUR", "0") != "0"


def _port() -> int:
    return int(_ADRESSE.rsplit(":", 1)[1])


def _url(chemin: str) -> str:
    return f"http://{_ADRESSE}{chemin}"


def _existe_serveur(timeout: float = 1.0) -> bool:
    """Un serveur repond deja sur le port ? (reuse si oui)"""
    try:
        with urllib.request.urlopen(_url("/health"), timeout=timeout) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def _binaire() -> Path | None:
    """Le binaire llama-server : AURA_SERVEUR_BIN, sinon serveur/llama-server.exe
    (recherche la version recente telechargee et les extractions datees)."""
    env = os.environ.get("AURA_SERVEUR_BIN")
    if env and Path(env).is_file():
        return Path(env)
    dossier = _RACINE / "serveur"
    if not dossier.is_dir():
        return None
    # preferer le plus recent (b11111 > b7400) : llama-server.exe direct ou
    # dans un sous-dossier
    candidats = sorted(dossier.rglob("llama-server.exe"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    return candidats[0] if candidats else None


def _gguf_cerveau() -> Path | None:
    """Le GGUF du cerveau : meme resolution que llama_cerveau._CHEMIN_GGUF."""
    env = os.environ.get("AURA_GGUF_CHEMIN")
    if env and Path(env).is_file():
        return Path(env)
    dossier = _RACINE / "modeles"
    if not dossier.is_dir():
        return None
    # Q8_0 prefere pour le serveur (pas de CPU_REPACK -> LoRA compatible),
    # sinon le premier GGUF Llama disponible
    for nom in ("Llama-3.2-1B-Instruct-Q8_0.gguf",
                "Llama-3.2-1B-Instruct-Q6_K.gguf",
                "Llama-3.2-1B-Instruct-Q4_K_M.gguf"):
        p = dossier / nom
        if p.is_file():
            return p
    ggufs = sorted(dossier.glob("*.gguf"), key=lambda p: p.stat().st_size)
    return ggufs[-1] if ggufs else None


def _adaptateurs_boot() -> list[Path]:
    """Les adaptateurs transposes (aura-*-T.gguf) du dossier serveur."""
    return sorted((_RACINE / "serveur").glob("aura-*-T.gguf"))


def _demarrer() -> bool:
    """Demarre le serveur avec le cerveau + les adaptateurs au boot."""
    global _process, _indispo
    bin_ = _binaire()
    gguf = _gguf_cerveau()
    if bin_ is None or gguf is None:
        _indispo = "binaire llama-server ou GGUF introuvable"
        LOG.info("[serveur_lora] %s -> chemin in-process", _indispo)
        return False
    cmd = [str(bin_), "-m", str(gguf), "--port", str(_port()),
           "--host", _ADRESSE.rsplit(":", 1)[0], "-c", "512", "-t", "4"]
    for p in _adaptateurs_boot():
        cmd += ["--lora", str(p)]
    try:
        _process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as e:
        _indispo = f"boot impossible : {e}"
        LOG.info("[serveur_lora] %s", _indispo)
        return False
    # attente de la sante (boot du 1B : ~5-15 s)
    for _ in range(60):
        if _process.poll() is not None:
            _indispo = f"serveur mort (code {_process.returncode})"
            LOG.info("[serveur_lora] %s", _indispo)
            return False
        if _existe_serveur():
            LOG.info("[serveur_lora] pret sur %s (%d adaptateurs)",
                     _ADRESSE, len(_adaptateurs_boot()))
            return True
        time.sleep(1.0)
    _indispo = "timeout de boot (20 s + 40 s d'attente)"
    LOG.info("[serveur_lora] %s", _indispo)
    return False


def _assurer() -> bool:
    """Serveur operationnel (demarre si besoin)."""
    if not _actifs():
        return False
    if _existe_serveur():
        return True
    if _indispo is not None:
        return False
    return _demarrer()


# ── hot-swap ────────────────────────────────────────────────────────────

def activer(categorie: str) -> bool:
    """Monte l'adaptateur de la categorie (scale 1.0, les autres a 0).

    Renvoie True si l'adaptateur est applique ; False sinon -> l'appelant
    continue sur le cerveau de base ou le chemin in-process.
    """
    global _actif
    if not _assurer():
        return False
    if categorie == _actif:
        return True        # deja actif : zero appel reseau
    cible = _IDS.get(categorie)
    if cible is None:
        # categorie sans adaptateur : tout desactiver
        corps = [{"id": i, "scale": 0.0} for i in _IDS.values()]
    else:
        corps = [{"id": i, "scale": (1.0 if i == cible else 0.0)}
                 for i in _IDS.values()]
    try:
        req = urllib.request.Request(
            _url("/lora-adapters"), method="POST",
            data=json.dumps(corps).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            ok = json.loads(r.read().decode()).get("success", False)
        if not ok:
            return False
        _actif = categorie
        LOG.info("[serveur_lora] adaptateur actif : %s", categorie or "(aucun)")
        return bool(cible is not None)
    except Exception as e:
        LOG.info("[serveur_lora] activation %s impossible : %s", categorie, e)
        return False


def chat(messages: list[dict], max_tokens: int = 256) -> str | None:
    """Generation via /v1/chat/completions (OpenAI-compatible).

    None = echec -> l'appelant retombe sur le chemin in-process.
    """
    if not _assurer():
        return None
    corps = {"messages": messages, "max_tokens": max_tokens,
             "temperature": 0.4, "top_p": 0.9,
             "stop": ["<|eot_id|>", "\nUtilisateur:", "\nUser:"]}
    try:
        req = urllib.request.Request(
            _url("/v1/chat/completions"), method="POST",
            data=json.dumps(corps).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            sortie = json.loads(r.read().decode())
        return (sortie.get("choices") or [{}])[0] \
            .get("message", {}).get("content", "").strip() or None
    except Exception as e:
        LOG.info("[serveur_lora] chat impossible : %s", e)
        return None


def disponible() -> bool:
    """Diagnostic externe : le chemin serveur est-il utilisable ?"""
    return _actifs() and _assurer()


def statut() -> dict:
    """Diagnostic : adresse, process, adaptateur actif, etat du serveur."""
    return {"actives": _actifs(), "adresse": _ADRESSE,
            "serveur_vivant": _existe_serveur(),
            "adaptateur": _actif,
            "indisponible": _indispo}


def arreter() -> None:
    """Arrete le serveur lance par Aura (laisse vivre un serveur externe)."""
    global _process, _actif
    if _process is not None:
        _process.terminate()
        _process = None
    _actif = None
