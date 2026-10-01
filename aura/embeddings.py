"""Encodeur local all-MiniLM-L6-v2 (ONNX INT8) : phrase -> vecteur 384-d.

Micro-modele ~23 Mo (modeles/minilm/, voir scripts/telecharger_minilm.py),
inference CPU ~2-5 ms par phrase — aucun LLM, aucune connexion runtime.

Trois composants partagent UNE session ONNX (init paresseuse, thread-safe) :
- aura/concepts.py            : detecteur de concepts abstraits ;
- aura/classificateur_flou.py : SVM sur vecteurs, aiguilleur de pensee ;
- filtre_instantane           : memoire semantique du cache de reponses.

Garanties :
- AURA_MINILM=0 -> encoder() renvoie None, chaque consommateur retombe sur
  son ancien chemin (TF-IDF / regles) : jamais d'erreur bloquante ;
- modele absent -> idem, avec un seul LOG.info explicite ;
- intra_op borne (2 threads) : l'encodeur ne vole pas les coeurs a llama.
"""
import logging
import os
import threading
from pathlib import Path

import numpy as np

LOG = logging.getLogger("aura.embeddings")

_DOSSIER = Path(__file__).resolve().parent.parent / "modeles" / "minilm"
_MODELE = _DOSSIER / "model_quantized.onnx"
_TOKENIZER = _DOSSIER / "tokenizer.json"

_DIM = 384
_MAX_JETONS = 256       # fenetre MiniLM (les questions sont bien plus courtes)
_THREADS = 2            # borne : llama/le serveur gardent les autres coeurs

_verrou = threading.Lock()
_session = None          # onnxruntime.InferenceSession | None
_tokenizer_obj = None    # tokenizers.Tokenizer | None
_indispo: str | None = None   # cause d'indisponibilite (une seule fois)


def actif() -> bool:
    """Interrupteur : AURA_MINILM != 0 (actif par defaut)."""
    return os.environ.get("AURA_MINILM", "1") != "0"


def modele_present() -> bool:
    return _MODELE.exists() and _TOKENIZER.exists()


def _initialiser() -> bool:
    """Ouvre tokenizer + session ONNX (une seule fois). False si absent/casse."""
    global _session, _tokenizer_obj, _indispo
    if _session is not None:
        return True
    if _indispo is not None or not actif():
        return False
    with _verrou:
        if _session is not None:
            return True
        if not actif():
            return False
        if not modele_present():
            _indispo = "modele absent"
            LOG.info("[minilm] modele absent -> embeddings desactives "
                     "(python scripts/telecharger_minilm.py pour activer "
                     "concepts/classificateur/cache semantique)")
            return False
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
            from tokenizers import Tokenizer
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = _THREADS
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _session = ort.InferenceSession(
                str(_MODELE), sess_options=opts,
                providers=["CPUExecutionProvider"])
            _tokenizer_obj = Tokenizer.from_file(str(_TOKENIZER))
            _tokenizer_obj.enable_truncation(max_length=_MAX_JETONS)
            # padding au PLUS LONG de lot (pas a 128 fixes : ~x8 de calcul evite)
            _tokenizer_obj.enable_padding(pad_id=0)
            LOG.info("[minilm] session ONNX prete (%s)",
                     ", ".join(_session.get_providers()))
            return True
        except Exception as e:  # noqa: BLE001
            _indispo = str(e)
            LOG.info("[minilm] init impossible -> embeddings desactives : %s", e)
            return False


def encoder(textes: list[str]) -> np.ndarray | None:
    """Vecteurs L2-normalises, shape (n, 384) — None si indisponible.

    Moyenne ponderee par le masque d'attention puis L2 : la norme du
    produit scalaire devient la similarite cosinus (0 multiplication
    par la suite, cache et classifieur = un simple dot).
    """
    if not textes:
        return np.zeros((0, _DIM), dtype=np.float32)
    if not _initialiser():
        return None
    try:
        assert _session is not None and _tokenizer_obj is not None
        enc = _tokenizer_obj.encode_batch([t or " " for t in textes])
        longueur = max(len(e.ids) for e in enc)
        ids = np.zeros((len(enc), longueur), dtype=np.int64)
        mask = np.zeros((len(enc), longueur), dtype=np.int64)
        # pooling SANS les tokens speciaux ([CLS]/[SEP]/[PAD]) : mesure
        # (12 paires requete->concept) = 11/12 bien classés contre 7/12
        # avec le pooling ST classique — le vecteur moyen des concepts
        # dominait la similarite et creait des « puits » attirant tout.
        spec = np.ones((len(enc), longueur), dtype=np.float32)
        for i, e in enumerate(enc):
            n = min(len(e.ids), longueur)
            ids[i, :n] = e.ids[:n]
            mask[i, :n] = e.attention_mask[:n]
            if getattr(e, "special_tokens_mask", None):
                spec[i, :n] = e.special_tokens_mask[:n]
        sorties = _session.run(
            ["last_hidden_state"],
            {"input_ids": ids, "attention_mask": mask,
             "token_type_ids": np.zeros_like(ids)})
        hidden = sorties[0].astype(np.float32)
        m = mask.astype(np.float32) * (1.0 - spec)
        vecs = ((hidden * m[:, :, None]).sum(axis=1)
                / np.maximum(m.sum(axis=1), 1e-9)[:, None])
        normes = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(normes, 1e-9)
    except Exception as e:  # noqa: BLE001
        LOG.warning("[minilm] encodage impossible : %s", e)
        return None


def encoder1(texte: str) -> np.ndarray | None:
    """Vecteur (384,) pour une seule phrase, ou None."""
    vecs = encoder([texte])
    if vecs is None or len(vecs) == 0:
        return None
    return vecs[0]


def moteur() -> str:
    """Diagnostic : « minilm » si actif, « absent » sinon."""
    if not actif():
        return "coupe"
    if _initialiser():
        return "minilm"
    return "absent"
