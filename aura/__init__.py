"""RontoGPT : IA hybride neuro-symbolique.

Trois piliers :
- orchestrateur : comprend la question, route, redige (LLM local, emplacement Mamba)
- expert_symbolique : Programmation Genetique Symbolique -> formules exactes (0 hallucination)
- memoire_web : RAG DuckDuckGo -> culture generale toujours a jour
"""
from .orchestrateur import Aura1B, RontoGPT

__all__ = ["RontoGPT", "Aura1B"]
__version__ = "0.1.0"
