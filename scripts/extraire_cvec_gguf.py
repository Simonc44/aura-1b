"""Extrait le vecteur de concept FR depuis le GGUF de production (llama.cpp).

Un callback cb_eval capture le tenseur « l_out-<COUCHE> » (sortie de la couche,
celle que llama_set_adapter_cvec modifie). Pour chaque paire (EN, FR) on prend
le dernier token ; le vecteur est la moyenne des deltas FR - EN.

Le vecteur ecrit va dans adaptateurs/langue_fr_heretic.cvec, via exporter_cvec.

Usage : PYTHONPATH=. python scripts/extraire_cvec_gguf.py
"""
import ctypes
import os
import sys

import numpy as np
import llama_cpp
from llama_cpp import llama_cpp as lc

from aura.llama_cerveau import chemin_gguf
from exporter_cvec import ecrire, relire
from intervention_tl import PAIRES_CONCEPT

COUCHE = 16
N_EMBD = 1152
NOM_TENSEUR = f"l_out-{COUCHE}".encode()
SORTIE = os.path.join("adaptateurs", "langue_fr_heretic.cvec")

_ggml = ctypes.CDLL(os.path.join(os.path.dirname(llama_cpp.__file__), "lib",
                                 "ggml-base.dll"))
_ggml.ggml_get_name.restype = ctypes.c_char_p
_ggml.ggml_get_name.argtypes = [ctypes.c_void_p]
_ggml.ggml_nbytes.restype = ctypes.c_size_t
_ggml.ggml_nbytes.argtypes = [ctypes.c_void_p]
_ggml.ggml_backend_tensor_get.restype = None
_ggml.ggml_backend_tensor_get.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]

_capture: dict = {}


def _callback(t, demande, _user):
    if _ggml.ggml_get_name(t) != NOM_TENSEUR:
        return False
    if demande:
        return True
    nb = _ggml.ggml_nbytes(t)
    assert nb % (4 * N_EMBD) == 0, "taille de tenseur inattendue"
    buf = (ctypes.c_float * (nb // 4))()
    _ggml.ggml_backend_tensor_get(t, buf, 0, nb)
    flat = np.frombuffer(buf, dtype=np.float32)
    _capture["h"] = flat[-N_EMBD:].copy()   # dernier token
    return True


CALLBACK = lc.ggml_backend_sched_eval_callback(_callback)


def _ouvrir(chemin: str):
    mp = lc.llama_model_default_params()
    mp.n_gpu_layers = 0
    modele = lc.llama_load_model_from_file(chemin.encode(), mp)
    cp = lc.llama_context_default_params()
    cp.n_ctx = 512
    cp.cb_eval = CALLBACK
    cp.cb_eval_user_data = None
    ctx = lc.llama_init_from_model(modele, cp)
    return modele, ctx


def _residu(modele, ctx, texte: str) -> np.ndarray:
    vocab = lc.llama_model_get_vocab(modele)
    brut = texte.encode("utf-8")
    cap = 512
    toks = (lc.llama_token * cap)()
    n = lc.llama_tokenize(vocab, brut, len(brut), toks, cap, True, False)
    assert n > 0, f"tokenisation echouee : {n}"
    lc.llama_memory_clear(lc.llama_get_memory(ctx), True)
    _capture.clear()
    batch = lc.llama_batch_get_one(toks, n)
    code = lc.llama_decode(ctx, batch)
    assert code == 0, f"llama_decode code {code}"
    assert "h" in _capture, f"tenseur {NOM_TENSEUR!r} non capture"
    return _capture["h"]


def main() -> int:
    chemin = chemin_gguf()
    print(f"cerveau : {chemin}")
    modele, ctx = _ouvrir(chemin)
    deltas = []
    for en, fr in PAIRES_CONCEPT:
        h_fr = _residu(modele, ctx, fr)
        h_en = _residu(modele, ctx, en)
        assert np.all(np.isfinite(h_fr)) and np.all(np.isfinite(h_en))
        deltas.append(h_fr - h_en)
        print(f"norme FR={np.linalg.norm(h_fr):.1f} EN={np.linalg.norm(h_en):.1f}")
    vec = np.mean(np.stack(deltas), axis=0).astype(np.float32)
    print(f"vecteur couche {COUCHE} : norme={np.linalg.norm(vec):.2f}")

    lc.llama_free(ctx)
    lc.llama_model_free(modele)

    ecrire(vec, 26, SORTIE)
    relu = relire(SORTIE)
    if not np.array_equal(relu[f"direction.{COUCHE}"], vec):
        print("ECHEC : relecture differente")
        return 1
    print(f"ecrit et verifie : {SORTIE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
