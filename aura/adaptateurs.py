"""Hot-swap d'adaptateurs LoRA : un cerveau, plusieurs specialites.

Principe : le GGUF de base est charge UNE fois. Les adaptateurs LoRA
(~5-40 Mo chacun, adapter GGUF produit par PEFT) sont montes/demontes
a la volee via la C-API llama.cpp, SANS recharger le modele. Chaque
categorie routee (web / math / code / general) peut avoir son
adaptateur specialise : le cerveau 1B change de specialite selon la
question, comme un MoE a faible cout.

Stockage : registre LRU en memoire (AURA_LRU, default 1 — seul
l'adaptateur actif est en RAM ; le surplus vit sur disque et se charge
au premier besoin). Cout memoire d'un adaptateur : quelques dizaines
de Mo au plus, souvent bien moins.

Surete de conception (meme doctrine que agents.py) :
- AURA_ADAPTATEURS=0 (default) ou aucun adaptateur installe -> ZERO
  surcout : le chemin par defaut est strictement le chemin d'avant ;
- tout echec (fichier absent, C-API refusee, version sans bindings) ->
  generation sans adaptateur + log, jamais bloquant ;
- pas de thread-lock (la suite est mono-thread ; le jour ou generer()
  sera multithread, ce module ajoutera le sien).

Production : scripts/entrainer_adaptateur.py (Colab, PEFT) -> 
adaptateurs/<nom>.gguf -> hot-swap automatique au routage.
"""
import ctypes
import importlib
import logging
import os
from collections import OrderedDict
from pathlib import Path

LOG = logging.getLogger("aura.adaptateurs")

# ── configuration ───────────────────────────────────────────────────────

_DOSSIER = Path(__file__).resolve().parent.parent / "adaptateurs"
# categorie routee -> nom de base du fichier adapter (sans extension)
_ADAPTATEURS = {
    "web": "aura-web",
    "math": "aura-math",
    "code": "aura-code",
    "general": "aura-general",
}
# LRU : nombre d'adaptateurs gardes en memoire (pointeurs charges).
# 1 = seul l'actif est en RAM : l'empreinte au repos est minimale.
_LRU_MAX = max(1, int(os.environ.get("AURA_LRU", "1")))

# ── etat ────────────────────────────────────────────────────────────────

# nom -> {"ptr": pointeur C, "path": Path} — OrderedDict = LRU
_REGISTRE: "OrderedDict[str, dict]" = OrderedDict()
_ACTIF: str | None = None        # adaptateur actuellement monte

_LIB = None                      # bindings C-API (charges paresseusement)
_ERR: str | None = None          # raison de l'indisponibilite (log 1 fois)


def _actifs() -> bool:
    """Interrupteur global (AURA_ADAPTATEURS=0 -> fonctionnalite OFF)."""
    return os.environ.get("AURA_ADAPTATEURS", "0") != "0"


def dossier() -> Path:
    """Dossier des adaptateurs (AURA_ADAPTATEURS_DOSSIER ou ./adaptateurs)."""
    d = os.environ.get("AURA_ADAPTATEURS_DOSSIER")
    return Path(d) if d else _DOSSIER


def _lib():
    """Bindings C-API llama.cpp ; None si la version n'expose pas le
    hot-swap (raison loggee une seule fois dans _ERR)."""
    global _LIB, _ERR
    if _LIB is not None or _ERR is not None:
        return _LIB
    try:
        c = importlib.import_module("llama_cpp.llama_cpp")
        requis = ("llama_adapter_lora_init", "llama_adapter_lora_free",
                  "llama_set_adapters_lora", "llama_adapter_lora_p")
        if all(hasattr(c, n) for n in requis):
            _LIB = c
        else:
            _ERR = "C-API lora absente de cette version de llama-cpp-python"
    except Exception as e:
        _ERR = f"llama_cpp.llama_cpp import impossible : {e}"
    if _ERR:
        LOG.info("[adaptateurs] hot-swap indisponible (%s) -> cerveau brut",
                 _ERR)
    return _LIB


def adaptateur_pour(categorie: str) -> Path | None:
    """Chemin de l'adaptateur de la categorie, s'il existe sur disque."""
    if not _actifs():
        return None
    nom = _ADAPTATEURS.get(categorie)
    if not nom:
        return None
    for ext in (".gguf", ".bin"):
        p = dossier() / f"{nom}{ext}"
        if p.is_file():
            return p
    return None


def _limiter() -> None:
    """LRU : libere les adaptateurs au-dela de _LRU_MAX (jamais l'actif,
    toujours le plus ancien d'abord)."""
    lib = _lib()
    while len(_REGISTRE) > _LRU_MAX:
        vieux, entree = _REGISTRE.popitem(last=False)
        if vieux == _ACTIF and len(_REGISTRE) < _LRU_MAX:
            # cas limite : l'actif serait evicté — on le garde
            _REGISTRE[vieux] = entree
            _REGISTRE.move_to_end(vieux)
            continue
        if lib is not None:
            try:
                lib.llama_adapter_lora_free(entree["ptr"])
            except Exception:
                pass
        LOG.info("[adaptateurs] LRU : %s libere de la memoire", vieux)


def _modele_brut(llm):
    """Pointeur C brut du modele (depaquette le wrapper LlamaModel)."""
    m = getattr(llm, "_model", None)
    inner = getattr(m, "model", None)  # wrapper Python -> pointeur ctypes
    return inner if inner is not None else m


def _ctx_brut(llm):
    """Pointeur C brut du contexte (depaquette le wrapper LlamaContext)."""
    c = getattr(llm, "_ctx", None)
    inner = getattr(c, "ctx", None)  # wrapper Python -> pointeur ctypes
    return inner if inner is not None else c


def _pointeur(llm, chemin: Path):
    """Pointeur C de l'adaptateur (charge ou deja en registre LRU)."""
    lib = _lib()
    if lib is None:
        return None
    nom = chemin.stem
    if nom in _REGISTRE:
        _REGISTRE.move_to_end(nom)
        return _REGISTRE[nom]["ptr"]
    try:
        ptr = lib.llama_adapter_lora_init(_modele_brut(llm), str(chemin).encode())
    except Exception as e:
        LOG.info("[adaptateurs] chargement %s impossible : %s", nom, e)
        return None
    if not ptr:
        LOG.info("[adaptateurs] %s refuse par llama.cpp (incompatible ?)",
                 nom)
        return None
    _REGISTRE[nom] = {"ptr": ptr, "path": chemin}
    _limiter()
    return _REGISTRE.get(nom, {}).get("ptr")


def appliquer(llm, categorie: str) -> bool:
    """Monte l'adaptateur de la categorie sur le contexte charge (hot-swap).

    Renvoie True si un adaptateur est effectivement applique ; False
    sinon (OFF, fichier absent, C-API indispo, echec) — dans tous les
    cas False = le cerveau de base continue tel quel.
    """
    global _ACTIF
    chemin = adaptateur_pour(categorie)
    if chemin is None:
        return False
    lib = _lib()
    if lib is None:
        return False
    try:
        ptr = _pointeur(llm, chemin)
        if ptr is None:
            return False
        elem = lib.llama_adapter_lora_p_ctypes          # POINTER(c_void_p)
        tableau = (elem * 1)(ctypes.cast(ptr, elem))    # llama_adapter_lora**
        echelles = (ctypes.c_float * 1)(1.0)
        lib.llama_set_adapters_lora(_ctx_brut(llm), tableau, 1, echelles)
        _ACTIF = chemin.stem
        LOG.info("[adaptateurs] actif : %s (hot-swap, scale 1.0)", _ACTIF)
        return True
    except Exception as e:
        LOG.info("[adaptateurs] application impossible (%s) -> cerveau brut",
                 e)
        return False


def desactiver(llm) -> None:
    """Retire l'adaptateur actif (retour au cerveau de base)."""
    global _ACTIF
    if _ACTIF is None:
        return
    lib = _lib()
    if lib is not None:
        try:
            lib.llama_set_adapters_lora(_ctx_brut(llm), None, 0, 0.0)
        except Exception as e:
            LOG.info("[adaptateurs] retrait impossible : %s", e)
    _ACTIF = None


def statut() -> dict:
    """Diagnostic : actif, registre LRU, dossier, interrupteur."""
    return {"actif": _ACTIF, "en_memoire": list(_REGISTRE),
            "lru_max": _LRU_MAX, "dossier": str(dossier()),
            "actives": _actifs()}
