"""Balayage des couches intermediaires du control vector FR (production).

Une seule passe llama.cpp capture l_out-<n> pour n dans COUCHES sur les 5
paires EN/FR (dernier token). Pour chaque couche, le vecteur de concept est
la moyenne des deltas FR - EN, applique seul (les autres couches a zero), puis
evalue sur les 10 questions tenues a l'ecart, comme eval_cvec_prod.py.

Usage : PYTHONPATH=. python scripts/balayage_couches_cvec.py
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

COUCHES = (10, 11, 12, 13, 14)
ECHELLES = (0.75, 1.0, 1.25)
N_EMBD = 1152
N_COUCHES = 26  # meme convention que exporter_cvec (layers 1..26)

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
    assert len(_capture) == len(COUCHES), f"capture incomplete : {sorted(_capture)}"
    return dict(_capture)


def extraire_concepts() -> dict:
    """{couche: vecteur moyen FR - EN} pour chaque couche de COUCHES."""
    chemin = chemin_gguf()
    mp = lc.llama_model_default_params()
    mp.n_gpu_layers = 0
    modele = lc.llama_load_model_from_file(chemin.encode(), mp)
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


def buffer_couche(couche: int, vec: np.ndarray) -> np.ndarray:
    """Buffer complet (layers 1..N_COUCHES-1 comme buf de test_cvec_prod)."""
    buf = np.zeros((N_COUCHES - 1) * N_EMBD, dtype=np.float32)
    buf[(couche - 1) * N_EMBD:couche * N_EMBD] = vec
    return buf


def main() -> int:
    print(f"cerveau : {chemin_gguf()}")
    concepts = extraire_concepts()
    for c in COUCHES:
        print(f"couche {c} : norme={np.linalg.norm(concepts[c]):.2f}")

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
        print(f"{nom:20s} justesse={ok}/{len(EVAL)} "
              f"fr_moy={np.mean(fr):.2f} pad={pads}", flush=True)

    evaluer("aucun", None)
    for c in COUCHES:
        for e in ECHELLES:
            evaluer(f"couche{c} x{e}", buffer_couche(c, concepts[c] * e))
    appliquer(llm, None, N_EMBD, N_COUCHES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
