"""Dynamic LoRA Layer Stepping (version control vector) sur le GGUF de production.

Le concept FR est extrait pour TOUTES les couches (l_out-1..25) en une passe
llama.cpp : moyenne des deltas FR - EN sur les 5 paires, dernier token.
L'amplitude appliquee a chaque couche suit une forme de rampe :

  - palier(a, b)       : 1 de a a b, 0 ailleurs
  - triangle(a, c, b)   : 0 en a, 1 en c, 0 en b, lineaire entre les deux

Le vecteur final est sum_L  w(L) * delta_L, ecrit dans le buffer par couche
attendu par llama_set_adapter_cvec (une couche = un segment de n_embd).

Usage : PYTHONPATH=. python scripts/rampe_couches_cvec.py
"""
import ctypes
import os
import sys

import numpy as np
import llama_cpp
from llama_cpp import Llama
from llama_cpp import llama_cpp as lc

from aura.llama_cerveau import chemin_gguf
from intervention_tl import PAIRES_CONCEPT
from eval_cvec_prod import EVAL, juste
from test_cvec_prod import appliquer, generer, score_langue

N_EMBD = 1152
N_COUCHES = 26           # convention exporter_cvec : layers 1..26
COUCHES = tuple(range(1, N_COUCHES))  # l_out-1 .. l_out-25

_ggml = ctypes.CDLL(os.path.join(os.path.dirname(llama_cpp.__file__), "lib",
                                 "ggml-base.dll"))
_ggml.ggml_get_name.restype = ctypes.c_char_p
_ggml.ggml_get_name.argtypes = [ctypes.c_void_p]
_ggml.ggml_nbytes.restype = ctypes.c_size_t
_ggml.ggml_nbytes.argtypes = [ctypes.c_void_p]
_ggml.ggml_backend_tensor_get.restype = None
_ggml.ggml_backend_tensor_get.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]

NOMS = {f"l_out-{c}".encode(): c for c in COUCHES}
_capture: dict = {}


def _callback(t, demande, _user):
    couche = NOMS.get(_ggml.ggml_get_name(t))
    if couche is None:
        return False
    if demande:
        return True
    nb = _ggml.ggml_nbytes(t)
    buf = (ctypes.c_float * (nb // 4))()
    _ggml.ggml_backend_tensor_get(t, buf, 0, nb)
    flat = np.frombuffer(buf, dtype=np.float32)
    _capture[couche] = flat[-N_EMBD:].copy()  # dernier token
    return True


CALLBACK = lc.ggml_backend_sched_eval_callback(_callback)


def _residus(modele, ctx, texte: str) -> dict:
    vocab = lc.llama_model_get_vocab(modele)
    brut = texte.encode("utf-8")
    cap = 512
    toks = (lc.llama_token * cap)()
    n = lc.llama_tokenize(vocab, brut, len(brut), toks, cap, True, False)
    assert n > 0, f"tokenisation echouee : {n}"
    lc.llama_memory_clear(lc.llama_get_memory(ctx), True)
    _capture.clear()
    code = lc.llama_decode(ctx, lc.llama_batch_get_one(toks, n))
    assert code == 0, f"llama_decode code {code}"
    assert len(_capture) == len(COUCHES), f"capture incomplete : {len(_capture)}"
    return dict(_capture)


def extraire_deltas() -> dict:
    """{couche: delta moyen FR - EN} pour chaque couche 1..25."""
    mp = lc.llama_model_default_params()
    mp.n_gpu_layers = 0
    modele = lc.llama_load_model_from_file(chemin_gguf().encode(), mp)
    cp = lc.llama_context_default_params()
    cp.n_ctx = 512
    cp.cb_eval = CALLBACK
    cp.cb_eval_user_data = None
    ctx = lc.llama_init_from_model(modele, cp)
    deltas = {c: [] for c in COUCHES}
    for en, fr in PAIRES_CONCEPT:
        r_fr = _residus(modele, ctx, fr)
        r_en = _residus(modele, ctx, en)
        for c in COUCHES:
            deltas[c].append(r_fr[c] - r_en[c])
    lc.llama_free(ctx)
    lc.llama_model_free(modele)
    return {c: np.mean(np.stack(d), axis=0).astype(np.float32)
            for c, d in deltas.items()}


def palier(a: int, b: int):
    return lambda L: 1.0 if a <= L <= b else 0.0


def triangle(a: int, c: int, b: int):
    def w(L):
        if L <= a or L >= b:
            return 0.0
        if L <= c:
            return (L - a) / (c - a)
        return (b - L) / (b - c)
    return w


K_NORME = 300.0  # norme commune imposee a chaque delta (energie egale)


def vecteur_rampe(deltas: dict, forme, echelle: float) -> np.ndarray:
    """Buffer complet : segment L-1 = echelle * w(L) * (K * delta_L / ||delta_L||).

    Normaliser chaque couche rend la forme de la rampe la seule variable :
    sans cela, les normes croissantes (172 a 1300) faussent la comparaison.
    """
    buf = np.zeros((N_COUCHES - 1) * N_EMBD, dtype=np.float32)
    for L in COUCHES:
        poids = echelle * forme(L)
        if poids != 0.0:
            d = deltas[L] / np.linalg.norm(deltas[L])
            buf[(L - 1) * N_EMBD:L * N_EMBD] = poids * K_NORME * d
    return buf


def main() -> int:
    print(f"cerveau : {chemin_gguf()}", flush=True)
    deltas = extraire_deltas()
    normes = {L: float(np.linalg.norm(deltas[L])) for L in COUCHES}
    print("normes 8..18 : " + " ".join(f"{L}:{normes[L]:.0f}" for L in range(8, 19)),
          flush=True)

    llm = Llama(model_path=chemin_gguf(), n_ctx=1024, verbose=False)

    def evaluer(nom: str, buf) -> None:
        if buf is not None:
            appliquer(llm, buf, N_EMBD, N_COUCHES)
        ok, fr, pads = 0, [], 0
        for question, attendu in EVAL:
            texte = generer(llm, question, max_tokens=24)
            ok += juste(texte, attendu)
            fr.append(score_langue(texte))
            pads += "<pad>" in texte
        print(f"{nom:24s} justesse={ok}/{len(EVAL)} "
              f"fr_moy={np.mean(fr):.2f} pad={pads}", flush=True)

    global K_NORME
    evaluer("aucun", None)
    for K_NORME in (100.0, 30.0):
        evaluer(f"K={K_NORME:.0f} palier 10-14", vecteur_rampe(deltas, palier(10, 14), 1.0))
        evaluer(f"K={K_NORME:.0f} triangle 8-12-16", vecteur_rampe(deltas, triangle(8, 12, 16), 1.0))
    appliquer(llm, None, N_EMBD, N_COUCHES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
