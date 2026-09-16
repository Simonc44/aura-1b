"""Orchestrateur autonome : AUCUNE dependance externe (pas Ollama, pas API).

Cerveau unique :
- Llama 3.2 1B Instruct (Q4_K_M, llama.cpp) : instruction-tuned, bon FR + EN

Plus :
- PGS (gplearn) : formules mathematiques exactes, erreur 0
- DuckDuckGo : faits en temps reel
- Auto-amelioration : corrections injectees dans le prompt

Le routeur (TF-IDF + LogReg) choisit les experts a activer ; Llama synthetise.
"""
import logging

from . import expert_symbolique, memoire_web, autoamelioration
from .routeur import RouteurIntelligent

LOG = logging.getLogger("aura.orchestrateur")


class Aura1B:
    """Systeme MoE autonome : cerveau Llama + experts PGS / web."""

    def __init__(self):
        self.expert = expert_symbolique.ExpertSymbolique()
        self.derniere_formule = ""
        self.derniere_erreur = None
        self._routeur = RouteurIntelligent()
        self._llama = None

    # -- routage intelligent ------------------------------------------------

    def analyser(self, question: str) -> dict:
        try:
            return self._routeur.classer(question)
        except Exception as e:
            LOG.warning("[routeur] fallback (%s)", e)
            return self._routeur.routeur_classique(question)

    # -- cerveau ------------------------------------------------------------

    def _generer(self, question: str, contexte_web: str, formule: str) -> str:
        if self._llama is None:
            from .llama_cerveau import generer as llama_generer
            self._llama = llama_generer
        return self._llama(question, contexte_web, formule)

    # -- prompt structure ---------------------------------------------------

    @staticmethod
    def _construire_prompt(question, contexte_web, formule) -> str:
        sections = [f"QUESTION : {question}"]
        ctx_corr = autoamelioration.construire_contexte_corrections(question)
        if ctx_corr:
            sections.append(f"\nCORRECTIONS :\n{ctx_corr}")
        if contexte_web:
            sections.append(f"\nWEB :\n{contexte_web}")
        if formule:
            sections.append(f"\nFORMULE : Y = {formule}")
        return "\n".join(sections)

    # -- pilier logique ----------------------------------------------------

    def resoudre_numerique(self, X, y) -> str:
        formule = self.expert.resoudre(X, y)
        self.derniere_formule = formule
        self.derniere_erreur = self.expert.erreur
        return formule

    # -- pipeline complet ---------------------------------------------------

    def executer(self, question: str, X=None, y=None) -> str:
        analyse = self.analyser(question)
        experts = analyse["experts"]
        contexte_web = memoire_web.chercher(question) if "web" in experts else ""
        formule = self.resoudre_numerique(X, y) if "math" in experts and X and y else ""
        return self._generer(question, contexte_web, formule)

    def executer_detaille(self, question: str, X=None, y=None) -> dict:
        analyse = self.analyser(question)
        experts = analyse["experts"]
        contexte_web = memoire_web.chercher(question) if "web" in experts else ""
        formule = self.resoudre_numerique(X, y) if "math" in experts and X and y else ""
        reponse = self._generer(question, contexte_web, formule)
        return {"question": question, "experts": experts, "analyse": analyse,
                "cerveau_choisi": "llama-3.2-1b", "contexte_web": contexte_web,
                "formule": formule, "erreur_pgs": self.derniere_erreur,
                "reponse": reponse}
