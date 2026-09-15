"""Aura-1B : IA hybride neuro-symbolique.

Trois piliers :
- orchestrateur : comprend la question, route, redige (LLM local, emplacement Mamba)
- expert_symbolique : Programmation Genetique Symbolique -> formules exactes (0 hallucination)
- memoire_web : RAG DuckDuckGo -> culture generale toujours a jour
"""
from .orchestrateur import Aura1B

__all__ = ["Aura1B"]
__version__ = "0.1.0"
