"""Orchestrateur RWKV-4 430M : expert generaliste (langage, culture, conversation).

RWKV est un RNN lineaire (pas un Transformer) : memoire a taille fixe,
vitesse constante, idéal pour le langage general. Contrairement a Mamba
specialise sur la logique, RWKV gere mieux la conversation et la culture.
Charge via le package `rwkv` (pur PyTorch, zero dependance CUDA).
"""
import logging
import os
import time

LOG = logging.getLogger("aura.rwkv")

_MODELE_PATH = None  # resolu au premier appel via huggingface_hub


class RWKVIndisponible(Exception):
    """Levee si RWKV ne peut pas tourner."""


class OrchestrateurRWKV:
    """Interface commune : generer(question, contexte_web, formule) -> str."""

    def __init__(self, max_tokens: int = 96):
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None

    def _resoudre_modele(self) -> str:
        """Trouve ou telecharge le .pth du modele RWKV-4 430M."""
        global _MODELE_PATH
        if _MODELE_PATH and os.path.exists(_MODELE_PATH):
            return _MODELE_PATH
        try:
            from huggingface_hub import hf_hub_download
            _MODELE_PATH = hf_hub_download(
                "BlinkDL/rwkv-4-pile-430m",
                "RWKV-4-Pile-430M-20220808-8066.pth",
            )
            return _MODELE_PATH
        except Exception as e:
            raise RWKVIndisponible(f"telechargement impossible : {e}") from e

    def _charger(self):
        if self._model is not None:
            return
        try:
            from rwkv.model import RWKV
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RWKVIndisponible(f"dependances manquantes : {e}") from e

        LOG.info("[rwkv] chargement du modele 430M ...")
        t0 = time.time()
        path = self._resoudre_modele()
        self._model = RWKV(model=path, strategy="cpu fp32")
        self._tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        LOG.info("[rwkv] pret en %.0fs", time.time() - t0)

    @property
    def disponible(self) -> bool:
        try:
            self._charger()
            return True
        except RWKVIndisponible:
            return False

    def generer(self, question: str, contexte_web: str, formule: str) -> str:
        """Redaction en langage general : conversation, culture, explications."""
        self._charger()
        import torch

        # prompt structure pour RWKV (base model, pas instruct)
        prompt = (
            f"### Human: {question}\n"
            f"### Assistant:"
        )
        tokens = self._tokenizer.encode(prompt)
        n_prompt = len(tokens)

        # premier appel : prompt complet -> state
        state = None
        x = torch.tensor([tokens]).long()
        out, state = self._model.forward(x, state)
        token = int(torch.argmax(out[-1]).item())
        tokens.append(token)

        # generation token par token
        for _ in range(self.max_tokens - 1):
            if token == 0:  # EOS
                break
            x = torch.tensor([[token]]).long()
            out, state = self._model.forward(x, state)
            token = int(torch.argmax(out[-1]).item())
            tokens.append(token)

        texte = self._tokenizer.decode(tokens[n_prompt:], skip_special_tokens=True)
        return texte.strip()
