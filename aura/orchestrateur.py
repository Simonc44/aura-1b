"""Orchestrateur autonome : AUCUNE dependance externe (pas Ollama, pas API).

Cerveau unique :
- Llama 3.2 1B Instruct (llama.cpp) : instruction-tuned, bon FR + EN,
  flash attention + KV cache q8_0 + memoire de conversation multi-tours

Plus :
- PGS (gplearn) : formules mathematiques exactes, erreur 0
- DuckDuckGo : faits en temps reel
- Auto-amelioration : corrections injectees dans le prompt

Le routeur (TF-IDF + LogReg) choisit les experts a activer ; Llama synthetise.
"""
import logging

from . import expert_symbolique, memoire_web, autoamelioration, filtre_instantane
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
        self._historique: list[dict] = []   # memoire de conversation

    # -- memoire de conversation -------------------------------------------

    def reinitialiser_conversation(self):
        """Oublie la conversation en cours (nouveau sujet)."""
        self._historique.clear()

    # -- routage intelligent ------------------------------------------------

    def analyser(self, question: str) -> dict:
        try:
            return self._routeur.classer(question)
        except Exception as e:
            LOG.warning("[routeur] fallback (%s)", e)
            return self._routeur.routeur_classique(question)

    # -- cerveau ------------------------------------------------------------

    def _generer(self, question: str, contexte_web: str, formule: str,
                 riche: bool = False) -> str:
        if self._llama is None:
            from .llama_cerveau import generer as llama_generer
            self._llama = llama_generer
        if riche:
            from .llama_cerveau import generer_riche
            return generer_riche(question, contexte_web, formule,
                                 historique=self._historique)
        return self._llama(question, contexte_web, formule,
                           historique=self._historique)

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

    # -- detection de complexite --------------------------------------------

    @staticmethod
    def _est_complexe(question: str) -> bool:
        """Les questions ouvertes meritent le mode riche (multi-pass, +style).

        Les questions factuelles (qui/ou/quand + fait precis) restent en
        mode simple : reponse courte rapide. Le mode riche coute 2 passes.
        """
        q = question.lower()
        if len(q.split()) >= 9:                      # question developpee
            return True
        return any(m in q for m in (
            "explique", "analyse", "compare", "pourquoi", "discute",
            "redige", "essai", "opinion", "avis", "argumente", "dissertation"))

    # -- pipeline complet ---------------------------------------------------

    def executer(self, question: str, X=None, y=None) -> str:
        return self.executer_detaille(question, X, y)["reponse"]

    def executer_detaille(self, question: str, X=None, y=None) -> dict:
        # ── NIVEAU 0 : reponse instantanee (0.1s) ─────────────────────
        # maths directes + cache semantique : la majorite des questions
        # quotidiennes n'ont PAS besoin du LLM (composant le plus lent).
        instant = filtre_instantane.repondre(question)
        if instant is not None and not (X and y):
            return {"question": question, "experts": {"instantane"},
                    "analyse": {"methodes": ["niveau0"]},
                    "cerveau_choisi": "niveau0-instantane",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": instant}

        # ── NIVEAUX 1-2 : experts + LLM ───────────────────────────────
        analyse = self.analyser(question)
        experts = analyse["experts"]
        riche = self._est_complexe(question)
        # mode riche : recherche enrichie (faits + analyses + lexique)
        if "web" in experts and riche:
            contexte_web = memoire_web.chercher_enrichi(question)
        else:
            contexte_web = memoire_web.chercher(question) if "web" in experts else ""
        formule = self.resoudre_numerique(X, y) if "math" in experts and X and y else ""
        reponse = self._generer(question, contexte_web, formule, riche=riche)
        # memorise pour les futures questions (cache semantique + conversation)
        filtre_instantane.enregistrer(question, reponse)
        self._historique.append({"role": "user", "content": question})
        self._historique.append({"role": "assistant", "content": reponse})
        return {"question": question, "experts": experts, "analyse": analyse,
                "cerveau_choisi": "llama-3.2-1b", "contexte_web": contexte_web,
                "formule": formule, "erreur_pgs": self.derniere_erreur,
                "reponse": reponse}
