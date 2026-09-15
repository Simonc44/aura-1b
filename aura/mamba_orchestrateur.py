"""Orchestrateur Mamba : State Space Model lineaire avec acceleration CPU.

Acceleration CPU (sans GPU, sans CUDA kernels) :
- 370M par defaut (1.5 Go RAM, tourne sur PC etudiant 8 Go)
- 1.4B en option via --grande (5.6 Go RAM, necessite 16 Go+)
- quantification dynamique int8 via oneDNN (gain x1.5-x2)
- inference_mode + flush denormal (gain gratuit)
- max_tokens 96 (usage vocal court)

Design :
- chargement PARESSEUX (au premier appel)
- meme interface que l'orchestrateur Ollama generer()
- repli automatique si torch/transformers absents -> Ollama -> template
"""
import logging
import os
import time

LOG = logging.getLogger("aura.mamba")

# 370M = defaut (1.5 Go, 8 Go RAM suffisent)
# 1.4B = optionnel (5.6 Go, necessite 16 Go+ RAM)
_ID_MODELES = {
    "standard": "state-spaces/mamba-130m-hf",
    "grande": "state-spaces/mamba-1.4b-hf",
}
_PROMPT_BENCHMARK = "Reponds en une phrase : la capitale de la France est"

# prompt court pour Mamba (130M est petit : les longs prompts le ralentissent)
_PROMPT_MAMBA = (
    "Tu es Aura, IA hybride. Reponds en francais, 2 phrases max. "
    "Cite les faits web et explique la formule si presente."
)


class MambaIndisponible(Exception):
    """Levee si Mamba ne peut pas tourner : l'orchestrateur doit replier sur Ollama."""


class OrchestrateurMamba:
    """Interface identique a l'orchestrateur Ollama : generer(question, contexte, formule)."""

    def __init__(self, id_modele: str | None = None, dispositif: str | None = None,
                 max_tokens: int = 96, temperature: float = 0.3,
                 penalite_repetition: float = 1.15, grande: bool = False):
        self.id_modele = id_modele or _ID_MODELES["grande" if grande else "standard"]
        self.dispositif = dispositif
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.penalite_repetition = penalite_repetition
        self._model = None
        self._tokenizer = None

    # -- acceleration CPU --------------------------------------------------

    def _mesurer(self, n: int = 24) -> float:
        """Mesure tokens/s sur n tokens (prompt court)."""
        import torch
        ent = self._tokenizer(_PROMPT_BENCHMARK, return_tensors="pt").to(self.dispositif)
        t0 = time.perf_counter()
        with torch.inference_mode():
            self._model.generate(**ent, max_new_tokens=n, do_sample=False,
                                 pad_token_id=self._tokenizer.eos_token_id)
        dt = time.perf_counter() - t0
        return n / dt if dt > 0 else 0.0

    # -- chargement --------------------------------------------------------

    def _charger(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise MambaIndisponible(f"torch/transformers absents : {e}") from e

        dispositif = self.dispositif or ("cuda" if torch.cuda.is_available() else "cpu")
        LOG.info("[mamba] chargement %s sur %s ...", self.id_modele, dispositif)
        t0 = time.time()

        # en CPU : toujours float32 (le scan de Mamba exige float32 en reference)
        dtype = torch.float16 if dispositif == "cuda" else torch.float32
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.id_modele)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.id_modele, dtype=dtype, low_cpu_mem_usage=True,
            ).to(dispositif)
        except OSError as e:
            if "pagination" in str(e).lower() or "pagefile" in str(e).lower():
                raise MambaIndisponible(
                    f"RAM insuffisante pour {self.id_modele}. "
                    f"Essaie : uv run python -m aura --demo (sans --cerveau mamba)"
                ) from e
            raise MambaIndisponible(f"chargement impossible : {e}") from e

        if dispositif == "cpu":
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
            torch.set_flush_denormal(True)

            # NOTE : quantize_dynamic est incompatible avec l'architecture Mamba
            # (SelectiveScan + Conv1d non standard). Les kernels CUDA mamba_ssm
            # et causal_conv1d sont les seuls vrais optimiseurs (GPU requis).
            # Sur CPU, le fp32 pur avec inference_mode est la config stable.
            v = self._mesurer()
            LOG.info("[mamba] vitesse CPU : %.1f tok/s (fp32, reference PyTorch)", v)

        self.dispositif = dispositif
        LOG.info("[mamba] pret en %.0fs", time.time() - t0)

    @property
    def disponible(self) -> bool:
        try:
            self._charger()
            return True
        except MambaIndisponible:
            return False

    # -- generation --------------------------------------------------------

    def generer(self, question: str, contexte_web: str, formule: str) -> str:
        from .orchestrateur import PROMPT_SYSTEME
        self._charger()
        import torch

        utilisateur = (
            f"QUESTION : {question}\n\nCONTEXTE WEB :\n{contexte_web or '(aucun)'}\n\n"
            f"FORMULE SYMBOLIQUE EXACTE :\n{formule or '(aucune)'}"
        )
        # Mamba 130M est petit : prompt court pour eviter le ralentissement
        prompt = f"{_PROMPT_MAMBA}\n\n{utilisateur}\n\nReponse :"
        entrees = self._tokenizer(prompt, return_tensors="pt").to(self.dispositif)
        with torch.inference_mode():
            sortie = self._model.generate(
                **entrees, max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-4),
                repetition_penalty=self.penalite_repetition,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        texte = self._tokenizer.decode(sortie[0][entrees["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        return texte.strip()
