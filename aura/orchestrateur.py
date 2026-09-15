"""Orchestrateur autonome : AUCUNE dépendance externe (pas Ollama, pas API).

Deux cerveaux locaux :
- Mamba 790M : logique, raisonnement, code (1.5 tok/s CPU)
- RWKV 430M : langage general, conversation (1.8 tok/s CPU)

Plus :
- PGS (gplearn) : formules mathematiques exactes, erreur 0
- DuckDuckGo : faits en temps reel
- Auto-amelioration : corrections injectees dans le prompt

Le systeme est plus capable que Qwen 1.5B SEUL car il combine :
- langage (Mamba/RWKV) + maths exactes (PGS) + web (DDG) + apprentissage
"""
import logging

from . import expert_symbolique, memoire_web, autoamelioration
from .routeur import RouteurIntelligent

LOG = logging.getLogger("aura.orchestrateur")

_PROMPT_MAMBA = (
    "Tu es Aura. Reponds en francais, 2 phrases max. "
    "Cite les faits web et explique la formule si presente."
)

_PROMPT_RWKV = (
    "### Human: {question}\n"
    "### Contexte web: {web}\n"
    "### Formule: {formule}\n"
    "### Assistant:"
)


class Aura1B:
    """Systeme MoE autonome : AUCUNE dépendance externe."""

    def __init__(self):
        self.expert = expert_symbolique.ExpertSymbolique()
        self.derniere_formule = ""
        self.derniere_erreur = None
        self._routeur = RouteurIntelligent()
        self._mamba = None
        self._rwkv = None

    # -- routage intelligent ------------------------------------------------

    def analyser(self, question: str) -> dict:
        try:
            return self._routeur.classer(question)
        except Exception as e:
            LOG.warning("[routeur] fallback (%s)", e)
            return self._routeur.routeur_classique(question)

    # -- selection du cerveau -----------------------------------------------

    def _choisir_cerveau(self, experts: set) -> str:
        if "math" in experts or "code" in experts:
            return "mamba"
        return "rwkv"

    # -- generation ---------------------------------------------------------

    def _generer(self, cerveau: str, question: str,
                 contexte_web: str, formule: str) -> str:
        if cerveau == "mamba":
            return self._generer_mamba(question, contexte_web, formule)
        return self._generer_rwkv(question, contexte_web, formule)

    def _generer_mamba(self, question, contexte_web, formule) -> str:
        if self._mamba is None:
            from .mamba_orchestrateur import OrchestrateurMamba
            self._mamba = OrchestrateurMamba()
        return self._mamba.generer(question, contexte_web, formule)

    def _generer_rwkv(self, question, contexte_web, formule) -> str:
        if self._rwkv is None:
            from .rwkv_orchestrateur import OrchestrateurRWKV
            self._rwkv = OrchestrateurRWKV()
        return self._rwkv.generer(question, contexte_web, formule)

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
        cerveau = self._choisir_cerveau(experts)
        return self._generer(cerveau, question, contexte_web, formule)

    def executer_detaille(self, question: str, X=None, y=None) -> dict:
        analyse = self.analyser(question)
        experts = analyse["experts"]
        contexte_web = memoire_web.chercher(question) if "web" in experts else ""
        formule = self.resoudre_numerique(X, y) if "math" in experts and X and y else ""
        cerveau = self._choisir_cerveau(experts)
        reponse = self._generer(cerveau, question, contexte_web, formule)
        return {"question": question, "experts": experts, "analyse": analyse,
                "cerveau_choisi": cerveau, "contexte_web": contexte_web,
                "formule": formule, "erreur_pgs": self.derniere_erreur,
                "reponse": reponse}
