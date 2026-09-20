"""Tests des adaptateurs LoRA (hot-swap) : sans llama.cpp reel (CI sans
GGUF, sans C-API de hot-swap selon versions), tout passe par des fakes.

Couvert :
- interrupteur AURA_ADAPTATEURS=0 : aucun chemin disque, aucun swap
- routage par categorie : bon fichier choisi, absent -> False
- registre LRU : eviction du plus ancien, libere via la C-API fake
- hot-swap : set_adapters_lora appele avec le bon pointeur + scale
- desactiver() : set_adapters_lora(None, 0)
- integration orchestrateur : _generer bascule vers la categorie routee
  et le failsafe couvre tout (exception -> generation normale)
"""
import ctypes

import pytest

from aura import adaptateurs
from aura.orchestrateur import Aura1B


class _FakeLib:
    """C-API llama.cpp minimale : compte les appels, pointeurs factices."""

    def __init__(self):
        self.inits, self.frees, self.sets = [], [], []

    # type fake du tableau de pointeurs : un vrai type ctypes (comme la
    # vraie C-API) pour que `(type * 1)(ptr)` fonctionne
    llama_adapter_lora_p = ctypes.POINTER(ctypes.c_void_p)

    def llama_adapter_lora_init(self, model, path):
        self.inits.append(path)
        # pointeur factice du MEME type que le tableau C (comme la vraie
        # API qui renvoie un llama_adapter_lora*)
        return ctypes.cast(
            ctypes.c_void_p(len(self.inits) + 1),
            ctypes.POINTER(ctypes.c_void_p))

    def llama_adapter_lora_free(self, ptr):
        self.frees.append(ptr)

    def llama_set_adapters_lora(self, ctx, tableau, n, scale):
        self.sets.append((tableau, n, scale))


def _llm():
    """Faux Llama : expose les attrs que le module utilise (_model/_ctx)."""
    return type("FauxLlama", (), {"_model": None, "_ctx": None})()


@pytest.fixture()
def lib_fake(monkeypatch):
    fake = _FakeLib()
    monkeypatch.setattr(adaptateurs, "_LIB", fake)
    monkeypatch.setattr(adaptateurs, "_ERR", None)
    monkeypatch.setattr(adaptateurs, "_lib", lambda: fake)
    adaptateurs._REGISTRE.clear()
    adaptateurs._ACTIF = None
    return fake


@pytest.fixture(autouse=True)
def propre(monkeypatch, tmp_path):
    """Etat propre : OFF par defaut, dossier isole, registre vide."""
    monkeypatch.delenv("AURA_ADAPTATEURS", raising=False)
    monkeypatch.delenv("AURA_LRU", raising=False)
    monkeypatch.setattr(adaptateurs, "_DOSSIER", tmp_path)
    monkeypatch.setattr(adaptateurs, "_REGISTRE", adaptateurs.OrderedDict())
    adaptateurs._ACTIF = None
    yield
    adaptateurs._REGISTRE.clear()
    adaptateurs._ACTIF = None


# ── interrupteur + routage ──────────────────────────────────────────────

class TestRoutage:
    def test_off_par_defaut(self, tmp_path):
        assert adaptateurs.adaptateur_pour("web") is None
        assert adaptateurs.statut()["actives"] is False

    def test_on_trouve_le_bon_fichier(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AURA_ADAPTATEURS", "1")
        f = tmp_path / "aura-math.gguf"
        f.write_bytes(b"x")
        assert adaptateurs.adaptateur_pour("math") == f
        assert adaptateurs.adaptateur_pour("web") is None

    def test_categorie_inconnue(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AURA_ADAPTATEURS", "1")
        (tmp_path / "aura-web.gguf").write_bytes(b"x")
        assert adaptateurs.adaptateur_pour("code") is None


# ── hot-swap + LRU ──────────────────────────────────────────────────────

class TestHotSwap:
    def test_applique_et_compte(self, monkeypatch, tmp_path, lib_fake):
        monkeypatch.setenv("AURA_ADAPTATEURS", "1")
        (tmp_path / "aura-math.gguf").write_bytes(b"x")
        assert adaptateurs.appliquer(_llm(), "math") is True
        # la C-API recoit le chemin en bytes
        assert lib_fake.inits == [(str(tmp_path / "aura-math.gguf")
                                   .encode())]
        assert lib_fake.sets and lib_fake.sets[-1][2] == 1.0
        assert adaptateurs._ACTIF == "aura-math"

    def test_echec_init_renvoie_false(self, monkeypatch, tmp_path,
                                      lib_fake):
        monkeypatch.setenv("AURA_ADAPTATEURS", "1")
        (tmp_path / "aura-code.gguf").write_bytes(b"x")
        lib_fake.llama_adapter_lora_init = \
            lambda m, p: 0      # llama.cpp refuse (ptr nul)
        assert adaptateurs.appliquer(_llm(), "code") is False
        assert adaptateurs._ACTIF is None

    def test_desactiver(self, monkeypatch, tmp_path, lib_fake):
        monkeypatch.setenv("AURA_ADAPTATEURS", "1")
        (tmp_path / "aura-web.gguf").write_bytes(b"x")
        adaptateurs.appliquer(_llm(), "web")
        adaptateurs.desactiver(_llm())
        assert adaptateurs._ACTIF is None
        assert lib_fake.sets[-1][:2] == (None, 0)

    def test_lru_evict_le_plus_ancien(self, monkeypatch, tmp_path,
                                      lib_fake):
        monkeypatch.setenv("AURA_ADAPTATEURS", "1")
        monkeypatch.setenv("AURA_LRU", "1")
        for nom in ("aura-web", "aura-math"):
            (tmp_path / f"{nom}.gguf").write_bytes(b"x")
            adaptateurs.appliquer(_llm(),
                                  nom.replace("aura-", ""))
        # LRU=1 : le premier a ete libere
        assert len(lib_fake.frees) == 1
        assert adaptateurs._ACTIF == "aura-math"


# ── C-API indisponible (versions anciennes) ─────────────────────────────

class TestSansCApi:
    def test_indispo_renvoie_false(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AURA_ADAPTATEURS", "1")
        (tmp_path / "aura-web.gguf").write_bytes(b"x")
        monkeypatch.setattr(adaptateurs, "_LIB", None)
        monkeypatch.setattr(adaptateurs, "_ERR", "pas de C-API")
        assert adaptateurs.appliquer(_llm(), "web") is False


# ── integration orchestrateur ───────────────────────────────────────────

class TestIntegration:
    @staticmethod
    def _neutraliser(monkeypatch):
        """Cache niveau 0 OFF (sinon la question ne quitte jamais le N0
        et n'atteint jamais le hot-swap)."""
        from aura import filtre_instantane
        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
        monkeypatch.setattr(filtre_instantane, "enregistrer",
                            lambda q, r: None)

    def test_generer_bascule_vers_la_categorie(self, monkeypatch,
                                               tmp_path, lib_fake):
        self._neutraliser(monkeypatch)
        monkeypatch.setenv("AURA_ADAPTATEURS", "1")
        (tmp_path / "aura-web.gguf").write_bytes(b"x")
        monkeypatch.setattr(adaptateurs, "_DOSSIER", tmp_path)
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: "ok")
        # un modele DEJA charge (le hot-swap ne charge jamais lui-meme)
        monkeypatch.setattr(llama_cerveau, "_llm", _llm())
        ia = Aura1B()
        ia.analyser = lambda q: {"experts": {"web"},
                                 "scores": {"web": 0.9},
                                 "methodes": ["t"]}
        assert ia.executer("prix de l or aujourd hui") == "ok"
        assert adaptateurs._ACTIF == "aura-web"

    def test_failsafe_exception_ne_bloque_pas(self, monkeypatch, tmp_path,
                                              lib_fake):
        self._neutraliser(monkeypatch)
        monkeypatch.setenv("AURA_ADAPTATEURS", "1")
        (tmp_path / "aura-math.gguf").write_bytes(b"x")
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: "ok")
        monkeypatch.setattr(llama_cerveau, "_llm", _llm())

        def boom(*a, **k):
            raise RuntimeError("C-API en panique")

        monkeypatch.setattr(adaptateurs, "appliquer", boom)
        ia = Aura1B()
        ia.analyser = lambda q: {"experts": {"math"},
                                 "scores": {"math": 0.9},
                                 "methodes": ["t"]}
        assert ia.executer("combien fait 12 fois 12") == "ok"

    def test_off_chemin_inchange(self, monkeypatch):
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: "ok")
        ia = Aura1B()
        ia.analyser = lambda q: {"experts": {"web"},
                                 "scores": {"web": 0.9},
                                 "methodes": ["t"]}
        assert ia.executer("prix de l or aujourd hui") == "ok"
