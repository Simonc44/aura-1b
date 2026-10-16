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

Production : scripts/colab_lora_gemma.ipynb (Colab autonome, PEFT) -> 
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


# ── COUPLAGE A L'ARCHITECTURE DU CERVEAU ────────────────────────────────
# Un adaptateur LoRA n'est PAS un plugin : c'est un DELTA DE POIDS calcule
# pour une architecture precise (meme nombre de couches, memes dimensions,
# memes tetes). Monte sur une autre base, il ne specialise rien — il produit
# du bruit, generalement SANS erreur visible.
#
# Le controle comparait jusqu'ici des CHAINES DE CARACTERES dans le nom du
# fichier (« llama-3.2-1b » in nom). Un fichier renomme passait donc, et un
# cerveau du bon type range sous un autre nom etait refuse : la garde
# protegeait exactement ceux qui n'en avaient pas besoin. Les nombres qui
# comptent sont DANS le GGUF et se lisent (aura/gguf_meta.py) : le verdict
# est desormais une MESURE.
#
# 9 oct. 2026 — BASCULE Llama-3.2 -> Gemma-3 puis RE-ENTRAINEMENT : le
# cerveau par defaut est Gemma (26 couches, dim 1152, 4/1 tetes) et les
# nouveaux adaptateurs sont calcules pour LUI. Le verdict ne repose plus
# sur une constante a resynchroniser a la main : il lit l'architecture des
# adaptateurs SUR LE DISQUE (leur en-tete GGUF porte `general.architecture`)
# et la compare a celle du cerveau charge (cf. verifier_cerveau).
#
# Ce que ça change : le jour ou l'on remplace les 4 fichiers .gguf par
# les versions Gemma, AUCUNE ligne de ce module n'a besoin de bouger — le
# montage s'active tout seul. Inversement, des adaptateurs Llama (ou d'une
# autre famille) restent refuses sur un cerveau Gemma, sans reglage.
# Production : scripts/colab_lora_gemma.ipynb (Colab).
#
# ARCHITECTURE_CIBLE reste la description de reference des adaptateurs
# ATTENDUS (ceux en cours de re-entrainement). Elle sert de repli quand un
# adaptateur est present mais son en-tete illisible, et de documentation ;
# elle n'est plus le critere du verdict.
ARCHITECTURE_CIBLE: dict[str, str | int] = {
    "famille": "gemma3",
    "parametres": "1B",
    "block_count": 26,
    "embedding_length": 1152,
    "head_count": 4,
    "head_count_kv": 1,
}

#: Architecture du cerveau MAINTENANT par defaut. Sert a documenter l'ecart
#: et a le mesurer, jamais a filtrer : le verdict vient du fichier.
ARCHITECTURE_CERVEAU: dict[str, str | int] = {
    "famille": "gemma3",
    "parametres": "1B",
    "block_count": 26,
    "embedding_length": 1152,
    "head_count": 4,
    "head_count_kv": 1,
}

#: Marqueurs de NOM utilises uniquement pour les messages et les dossiers
#: d'adaptateurs par famille (jamais pour autoriser ou refuser un montage :
#: cf. verifier_cerveau, qui lit le GGUF).
MOTIFS_CERVEAU: tuple[str, ...] = ("llama-3.2-1b", "llama-3-2-1b", "llama3.2-1b")
MOTIFS_GEMMA: tuple[str, ...] = ("gemma-3-1b", "gemma3-1b", "gemma-3-1b-it")


def cerveau_compatible(chemin: str | Path) -> bool:
    """Le cerveau donne est-il celui pour lequel les LoRA ont ete calcules ?

    Lit l'EN-TETE du GGUF (aura/gguf_meta.py) : nombre de couches, dimension,
    tetes. Un nom de fichier ne prouve rien — un fichier renomme passait la
    version precedente de ce controle.
    """
    ok, _ = verifier_cerveau(chemin)
    return ok


def adaptateurs_installes() -> list[Path]:
    """Chemin de chaque adaptateur present sur disque (toutes categories)."""
    d = dossier()
    if not d.is_dir():
        return []
    trouves: list[Path] = []
    for nom in _ADAPTATEURS.values():
        for ext in (".gguf", ".bin"):
            p = d / f"{nom}{ext}"
            if p.is_file():
                trouves.append(p)
                break
    return trouves


def architecture_adaptateurs() -> dict:
    """Famille (et champs disponibles) lus dans les adaptateurs sur disque.

    Un LoRA GGUF porte `general.architecture` (mesure sur les fichiers du
    projet : `llama`), mais PAS `block_count`/`embedding_length` — ces cles
    n'existent que dans le GGUF du cerveau. On agrège donc ce qui EST la.
    """
    from . import gguf_meta

    familles: set[str] = set()
    for p in adaptateurs_installes():
        f = gguf_meta.famille(p)
        if f:
            familles.add(f)
    return {"familles": sorted(familles)}


def verifier_cerveau(chemin: str | Path) -> tuple[bool, str]:
    """(compatible, message). A appeler AVANT de monter des adaptateurs.

    Le verdict compare l'architecture RELLE des adaptateurs PRESENTS sur
    disque a celle du cerveau charge — plus une constante a tenir a jour.
    Remplacer les .gguf par les versions Gemma suffit donc a activer le
    montage. Ne leve jamais : renvoie un verdict que l'appelant journalise
    ou refuse.
    """
    from . import gguf_meta

    arc_cerveau = gguf_meta.architecture(chemin)
    if not arc_cerveau:
        return False, (f"architecture du cerveau "
                       f"'{Path(str(chemin)).name}' illisible "
                       "(fichier absent ou tronque)")
    installes = adaptateurs_installes()
    if not installes:
        return True, ""              # aucun adaptateur : rien a proteger
    for p in installes:
        arc = gguf_meta.architecture(p)
        if not arc:                  # en-tete illisible : llama.cpp tranchera
            continue
        ecarts = []
        if arc.get("famille") != arc_cerveau.get("famille"):
            ecarts.append(f"famille : adaptateur {arc.get('famille')}, "
                          f"cerveau {arc_cerveau.get('famille')}")
        # Ne comparer un nombre que si l'ADAPTATEUR l'expose : les LoRA
        # GGUF ne portent pas block_count, exiger l'egalite contre None
        # refuserait un adaptateur pourtant correct.
        for cle in ("block_count", "embedding_length", "head_count",
                    "head_count_kv"):
            vu, attendu = arc.get(cle), arc_cerveau.get(cle)
            if vu is not None and attendu is not None and vu != attendu:
                ecarts.append(f"{cle} : adaptateur {vu}, cerveau {attendu}")
        if ecarts:
            return False, (
                f"les adaptateurs installes ne sont pas ceux du cerveau "
                f"a monter — {'; '.join(ecarts)} "
                f"({gguf_meta.resume(chemin)}). Un LoRA est un delta de "
                "poids pour une architecture precise ; monte sur une autre "
                "base il produit du bruit SANS erreur visible. Ils sont "
                "ignores en attendant leur re-entrainement "
                "(scripts/colab_lora_gemma.ipynb).")
    return True, ""


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
        libere = True
        if lib is not None:
            try:
                lib.llama_adapter_lora_free(entree["ptr"])
            except Exception as e:
                # La memoire de l'adaptateur reste attachee : le log d'info
                # ci-dessous ne doit PAS annoncer une liberation qui n'a pas
                # eu lieu.
                libere = False
                LOG.warning("[adaptateurs] liberation de %s impossible (%s) : "
                            "memoire de l'adaptateur conservee", vieux, e)
        if libere:
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
