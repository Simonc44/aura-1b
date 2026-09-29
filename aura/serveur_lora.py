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
import threading
import time
from pathlib import Path

import urllib.request

LOG = logging.getLogger("aura.serveur_lora")

_RACINE = Path(__file__).resolve().parent.parent

# categorie routee -> id de l'adaptateur au boot du serveur
_IDS = {"math": 0, "web": 1, "code": 2, "general": 3}

_ADRESSE = os.environ.get("AURA_SERVEUR_ADRESSE", "127.0.0.1:8757")

_process: subprocess.Popen | None = None
_actif: tuple | None = None        # mix actuellement applique ((cat, scale)...)
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


_aides: dict[str, str] = {}


def _aide_binaire(bin_: Path) -> str:
    """Texte de --help du binaire (mis en cache, 1 probe au maximum)."""
    cle = str(bin_)
    if cle not in _aides:
        try:
            h = subprocess.run([str(bin_), "--help"], capture_output=True,
                               text=True, timeout=15)
            _aides[cle] = h.stdout + h.stderr
        except Exception:
            _aides[cle] = ""
    return _aides[cle]


def _vulkan() -> Path | None:
    """Build officiel win-vulkan de llama.cpp, si present et non desactive.

    Installable par scripts/installer_vulkan.ps1 (zip bXXXX officiel).
    Mesure sur ce PC (scripts/bench_gpu.py, Llama-3.2-1B Q4_K_M, 256 tok) :
        CPU     : decode 18,6-19,0 tok/s | prefill  66-73 tok/s
        VULKAN  : decode 20,4-20,6 tok/s | prefill 216-227 tok/s (x3,1)
    Le prefill est ce qui compte : chaque passe du multi-pass re-traite son
    prompt. AURA_GPU=0 force le chemin CPU.
    """
    if os.environ.get("AURA_GPU") == "0":
        return None
    exe = _RACINE / "serveur" / "vulkan" / "llama-server.exe"
    if exe.is_file() and (exe.parent / "ggml-vulkan.dll").is_file():
        return exe
    return None


def _binaire() -> Path | None:
    """Le binaire llama-server : AURA_SERVEUR_BIN, sinon serveur/llama-server.exe
    (recherche la version recente telechargee et les extractions datees)."""
    env = os.environ.get("AURA_SERVEUR_BIN")
    if env and Path(env).is_file():
        return Path(env)
    vulkan = _vulkan()
    if vulkan is not None:
        return vulkan
    dossier = _RACINE / "serveur"
    if not dossier.is_dir():
        return None
    # Preferer le plus recent (b11111 > b7400) : llama-server.exe direct ou
    # dans un sous-dossier. Le tri mtime seul se trompe ici (serveur/b7400/
    # extraite APRES la racine donc mtime plus jeune) — or ce b7400 est un
    # ancien build qui plante avec les 4 LoRA (GGML_ASSERT au 1er token, cf.
    # scripts/diag_serveur_crash.py) et ignore --spec-type. Signal de
    # version fiable : la presence de --spec-type dans le --help.
    candidats = sorted(dossier.rglob("llama-server.exe"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidats:
        return None
    for c in candidats:
        if "--spec-type" in _aide_binaire(c):
            return c
    return candidats[0]


def _gguf_cerveau() -> Path | None:
    """Le GGUF du cerveau : meme resolution que llama_cerveau._CHEMIN_GGUF.

    Q4_K_M PAR DEFAUT (choix utilisateur : 807 Mo, base de reference du
    projet, compatible LoRA une fois les adaptateurs transposes). Q8_0
    en variante fidele si Q4 absent.
    """
    env = os.environ.get("AURA_GGUF_CHEMIN")
    if env and Path(env).is_file():
        return Path(env)
    dossier = _RACINE / "modeles"
    if not dossier.is_dir():
        return None
    for nom in ("Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                "Llama-3.2-1B-Instruct-Q6_K.gguf",
                "Llama-3.2-1B-Instruct-Q8_0.gguf"):
        p = dossier / nom
        if p.is_file():
            return p
    ggufs = sorted(dossier.glob("*.gguf"), key=lambda p: p.stat().st_size)
    return ggufs[-1] if ggufs else None


def _adaptateurs_boot() -> list[Path]:
    """Les adaptateurs transposes (aura-*-T.gguf) du dossier serveur."""
    return sorted((_RACINE / "serveur").glob("aura-*-T.gguf"))


_spec_supporte: dict[str, bool] = {}


def _spec_type_ok(bin_: Path) -> bool:
    """Le binaire accepte-t-il --spec-type (speculative decoding n-gram) ?

    Le dossier sert/ peut contenir plusieurs versions (b7400 sans spec,
    b11111 avec) : on sonde --help une fois par binaire pour ne pas faire
    crasher un ancien. AURA_SERVEUR_SPEC=0 coupe tout (AURA_SPEC ne
    concerne QUE le chemin in-process, qui lui ralentit — cf.
    llama_cerveau._draft_model).

    Bench (scripts/bench_spec_serveur.py, Llama-3.2-1B Q4_K_M, 6 coeurs) :
        sans spec          19 tok/s   (repétitif)  19 tok/s   (neuf)
        ngram-cache        38 tok/s x2 (repétitif) 19.7 tok/s (neuf, neutre)
        ngram-mod          37 tok/s x2 (repétitif) 17 tok/s   (neuf, -10%)
        ngram-simple/map-* ~1.2-1.5x  mais -20% sur du texte neuf
    -> ngram-cache : gain net partout, aucun checkpoint draft a telecharger
    (Medusa/EAGLE-3 : inexistants pour Llama-3.2-1B).
    """
    if os.environ.get("AURA_SERVEUR_SPEC", "1") == "0":
        return False
    cle = str(bin_)
    if cle not in _spec_supporte:
        aide = _aide_binaire(bin_)
        _spec_supporte[cle] = ("--spec-type" in aide and "ngram-cache" in aide)
        LOG.info("[serveur_lora] speculative decoding (ngram-cache) : %s",
                 "actif" if _spec_supporte[cle] else "non supporte par ce binaire")
    return _spec_supporte[cle]


def _demarrer() -> bool:
    """Demarre le serveur avec le cerveau + les adaptateurs au boot."""
    global _process, _indispo
    bin_ = _binaire()
    gguf = _gguf_cerveau()
    if bin_ is None or gguf is None:
        _indispo = "binaire llama-server ou GGUF introuvable"
        LOG.info("[serveur_lora] %s -> chemin in-process", _indispo)
        return False
    # -c 1536 : le warmup avec 4 LoRA montes consomme ~10x la memoire de
    # contexte d'un boot simple (GGML_ASSERT sinon) ; --no-warmup en plus.
    cmd = [str(bin_), "-m", str(gguf), "--port", str(_port()),
           "--host", _ADRESSE.rsplit(":", 1)[0], "-c", "1536", "-t", "4",
           "--no-warmup"]
    if _spec_type_ok(bin_):
        # brouillon n-gram (cache) extrait de l'historique : x2 sur du texte
        # repetitif, cout nul sur du texte neuf — bench scripts/bench_spec_serveur.py
        cmd += ["--spec-type", "ngram-cache"]
    adaptateurs = _adaptateurs_boot()
    # Build GPU (Vulkan) : tous les layers sur l'iGPU (repli CPU automatique
    # par llama.cpp si aucun device — jamais de plantage).
    if _vulkan() == bin_:
        cmd += ["-ngl", "99"]
    if "--spec-type" in _aide_binaire(bin_):
        # build recent : --lora accepte une liste separee par des virgules
        # (les occurrences repetees declenchent un warning deprecation).
        if adaptateurs:
            cmd += ["--lora", ",".join(str(p) for p in adaptateurs)]
    else:
        for p in adaptateurs:
            cmd += ["--lora", str(p)]
    try:
        journal = open(_RACINE / "serveur" / "autogere.log",
                       "ab", buffering=0)
        _process = subprocess.Popen(
            cmd, stdout=journal, stderr=journal, cwd=str(_RACINE),
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

# Verrou global : serialise hot-swap (POST /lora-adapters) et generation
# (/v1/chat/completions). Un swap pendant une generation modifie les
# pointeurs de tenseurs en memoire C de llama.cpp sous ses pieds —
# resultat : sortie corrompue (melange de specialites) ou crash.
_verrou = threading.Lock()

def activer(categorie: str) -> bool:
    """Monte l'adaptateur de la categorie seul (API simple = mix 100/0)."""
    return activer_mixte({categorie: 1.0} if categorie else {})


def activer_mixte(poids: dict[str, float]) -> bool:
    """Monte PLUSIEURS adaptateurs simultanement (multi-LoRA a taille
    constante) : poids = {categorie: echelle 0..1}.

    Ex : {"math": 0.7, "web": 0.3} — une question qui touche deux sujets
    profite des deux specialites, sans recharger quoi que ce soit (le
    serveur ne change que les echelles, cout ~0).
    Renvoie True si le serveur a accepte.

    SECURITE THREADS : le verrou serialise POST /lora-adapters vs
    /v1/chat/completions — sans lui, une generation en cours pendant le
    swap des pointeurs de tenseurs en memoire C peut produire une sortie
    corrompue (melange de deux specialites) ou un crash du serveur.
    """
    global _actif
    cle = tuple(sorted((k, round(v, 3)) for k, v in poids.items() if v > 0))
    if cle == _actif:
        return True        # deja applique : zero appel reseau
    with _verrou:          # aucun chat() pendant la modification des echelles
        if cle == _actif:  # double-check : un autre thread a deja applique
            return True
        if not _assurer():
            return False
        corps = [{"id": i, "scale": round(float(poids.get(cat, 0.0)), 3)}
                 for cat, i in _IDS.items()]
        try:
            req = urllib.request.Request(
                _url("/lora-adapters"), method="POST",
                data=json.dumps(corps).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                ok = json.loads(r.read().decode()).get("success", False)
            if not ok:
                return False
            _actif = cle
            LOG.info("[serveur_lora] mix actif : %s",
                     dict(cle) if cle else "(cerveau brut)")
            return True
        except Exception as e:
            LOG.info("[serveur_lora] mix impossible : %s", e)
            return False


def chat(messages: list[dict], max_tokens: int = 256) -> str | None:
    """Generation via /v1/chat/completions (OpenAI-compatible).

    None = echec -> l'appelant retombe sur le chemin in-process.
    Le verrou garantit qu'aucun hot-swap ne modifie les echelles des
    adaptateurs PENDANT la generation (sinon : sortie corrompue).
    """
    if not _assurer():
        return None
    corps = {"messages": messages, "max_tokens": max_tokens,
             "temperature": 0.4, "top_p": 0.9,
             "stop": ["<|eot_id|>", "\nUtilisateur:", "\nUser:"]}
    try:
        with _verrou:  # vs activer_mixte() : jamais les deux a la fois
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
    """Diagnostic : adresse, process, mix actif, etat du serveur."""
    return {"actives": _actifs(), "adresse": _ADRESSE,
            "serveur_vivant": _existe_serveur(),
            "adaptateur": dict(_actif) if _actif else None,
            "indisponible": _indispo}


def arreter() -> None:
    """Arrete le serveur lance par Aura (laisse vivre un serveur externe)."""
    global _process, _actif
    if _process is not None:
        _process.terminate()
        _process = None
    _actif = None
