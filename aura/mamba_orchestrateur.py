"""Orchestrateur Mamba : State Space Model lineaire en remplacement du LLM Transformer.

Pourquoi Mamba : memoire de taille FIXE (que l'entree fasse 1 phrase ou 500 pages),
vitesse de generation constante, RAM maitrisee — ideal sur CPU.

Modele : state-spaces/mamba-1.3b-hf (port HF officiel, format transformers >= 4.46).
Design :
- chargement PARESSEUX (au premier appel) : importer le module ne coute rien
- torch par defaut sur CPU + threads optimaux ; CUDA si dispo
- si torch/transformers absents ou modele indisponible -> MambaIndisponible,
  et Aura1B retombe automatiquement sur Ollama (aucun blocage)
"""
import logging
import os

LOG = logging.getLogger("aura.mamba")

_ID_MODELE = "state-spaces/mamba-1.4b-hf"


class MambaIndisponible(Exception):
    """Levee si Mamba ne peut pas tourner : l'orchestrateur doit replier sur Ollama."""


class OrchestrateurMamba:
    """Interface identique a l'orchestrateur Ollama : generer(question, contexte, formule)."""

    def __init__(self, id_modele: str = _ID_MODELE, dispositif: str | None = None,
                 max_tokens: int = 256, temperature: float = 0.3,
                 penalite_repetition: float = 1.15):
        self.id_modele = id_modele
        self.dispositif = dispositif  # None = auto (cuda si dispo, sinon cpu)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.penalite_repetition = penalite_repetition
        self._model = None
        self._tokenizer = None

    def _charger(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise MambaIndisponible(f"torch/transformers absents : {e}") from e

        dispositif = self.dispositif or ("cuda" if torch.cuda.is_available() else "cpu")
        LOG.info("chargement de %s sur %s ...", self.id_modele, dispositif)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.id_modele)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.id_modele, torch_dtype=torch.float32 if dispositif == "cpu" else torch.float16,
            ).to(dispositif)
        except Exception as e:
            raise MambaIndisponible(f"chargement impossible : {e}") from e
        if dispositif == "cpu":
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        self.dispositif = dispositif

    @property
    def disponible(self) -> bool:
        try:
            self._charger()
            return True
        except MambaIndisponible:
            return False

    def generer(self, question: str, contexte_web: str, formule: str) -> str:
        from .orchestrateur import PROMPT_SYSTEME  # prompt unique, une seule source de verite
        self._charger()
        import torch

        utilisateur = (
            f"QUESTION : {question}\n\nCONTEXTE WEB :\n{contexte_web or '(aucun)'}\n\n"
            f"FORMULE SYMBOLIQUE EXACTE :\n{formule or '(aucune)'}"
        )
        prompt = f"{PROMPT_SYSTEME}\n\n{utilisateur}\n\nReponse :"
        entrees = self._tokenizer(prompt, return_tensors="pt").to(self.dispositif)
        with torch.no_grad():
            sortie = self._model.generate(
                **entrees, max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-4),
                repetition_penalty=self.penalite_repetition,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        texte = self._tokenizer.decode(sortie[0][entrees["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        return texte.strip()
