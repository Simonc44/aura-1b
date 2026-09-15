"""Orchestrateur MoE : comprend, route, active les experts, synthetise.

Trois cerveaux specialises :
- Mamba 790M : logique, raisonnement, code
- RWKV 430M : langage general, conversation, culture
- Ollama 1.5B : repli universel (qualite de langue)

Routeur intelligent : TF-IDF + LogReg (150 exemples) + cosinus prototypes.
Auto-amelioration : les corrections utilisateur sont injectees dans le prompt.
"""
import logging

import requests

from . import expert_symbolique, memoire_web, autoamelioration
from .routeur import RouteurIntelligent

LOG = logging.getLogger("aura.orchestrateur")

_PROMPT_SYSTEME = (
    "Tu es Aura, une IA neuro-symbolique. Tu recois 4 sections distincies :\n"
    "- CORRECTIONS ANTERIEURES : erreurs passees et leur correction\n"
    "- CONTEXTE WEB : faits verifies recuperes sur internet\n"
    "- FORMULE EXACTE : une loi mathematique precise\n"
    "- QUESTION : ce que l'utilisateur demande\n\n"
    "CONSIGNES :\n"
    "1. Si des corrections existent, NE RECOMAINE PAS l'erreur.\n"
    "2. Si une formule est presente, explique-la simplement.\n"
    "3. Si le contexte web contient des faits, integre-les.\n"
    "4. Ne JAMAIS inventer de faits non fournis.\n"
    "5. Reponds en 3 phrases maximum, claires et directes."
)


class Aura1B:
    """Chef d'orchestre MoE des quatre piliers."""

    def __init__(self, hote="http://localhost:11434", modele="qwen2.5:1.5b-instruct",
                 timeout=90, cerveau="ollama", mamba_grande=False):
        self.hote = hote.rstrip("/")
        self.modele = modele
        self.timeout = timeout
        self.expert = expert_symbolique.ExpertSymbolique()
        self.derniere_formule = ""
        self.derniere_erreur = None
        self._routeur = RouteurIntelligent()
        # cerveaux
        self._mamba = None
        self._rwkv = None
        self._cerveau_defaut = cerveau
        self._mamba_grande = mamba_grande
        self._mamba_actif = (cerveau == "mamba")
        self._rwkv_actif = (cerveau == "rwkv")

    # -- routage intelligent ------------------------------------------------

    def analyser(self, question: str) -> dict:
        """Analyse la question et decide quels experts activer."""
        try:
            return self._routeur.classer(question)
        except Exception as e:
            LOG.warning("[routeur] echec (%s), fallback mots-cles", e)
            return self._routeur.routeur_classique(question)

    # -- selection du cerveau -----------------------------------------------

    def _choisir_cerveau(self, experts: set) -> str:
        """Choisit le cerveau optimal selon les experts requis."""
        if "math" in experts or "code" in experts:
            return "mamba"
        if "general" in experts:
            return "rwkv"
        if "web" in experts:
            return "rwkv"
        return self._cerveau_defaut

    def _generer_cerveau(self, cerveau: str, question: str,
                         contexte_web: str, formule: str) -> str:
        """Genere via le cerveau selectionne."""
        if cerveau == "mamba":
            return self._generer_mamba(question, contexte_web, formule)
        elif cerveau == "rwkv":
            return self._generer_rwkv(question, contexte_web, formule)
        else:
            return self._generer_ollama(question, contexte_web, formule)

    def _generer_mamba(self, question, contexte_web, formule) -> str:
        if self._mamba is None:
            try:
                from .mamba_orchestrateur import OrchestrateurMamba
                self._mamba = OrchestrateurMamba(grande=self._mamba_grande)
            except Exception as e:
                LOG.warning("Mamba indisponible (%s)", e)
                return self._generer_ollama(question, contexte_web, formule)
        try:
            return self._mamba.generer(question, contexte_web, formule)
        except Exception as e:
            LOG.warning("Mamba erreur (%s), repli Ollama", e)
            return self._generer_ollama(question, contexte_web, formule)

    def _generer_rwkv(self, question, contexte_web, formule) -> str:
        if self._rwkv is None:
            try:
                from .rwkv_orchestrateur import OrchestrateurRWKV
                self._rwkv = OrchestrateurRWKV()
            except Exception as e:
                LOG.warning("RWKV indisponible (%s)", e)
                return self._generer_ollama(question, contexte_web, formule)
        try:
            return self._rwkv.generer(question, contexte_web, formule)
        except Exception as e:
            LOG.warning("RWKV erreur (%s), repli Ollama", e)
            return self._generer_ollama(question, contexte_web, formule)

    def _generer_ollama(self, question, contexte_web, formule) -> str:
        utilisateur = self._construire_prompt(question, contexte_web, formule)
        try:
            r = requests.post(f"{self.hote}/api/chat", json={
                "model": self.modele, "stream": False,
                "messages": [{"role": "system", "content": _PROMPT_SYSTEME},
                             {"role": "user", "content": utilisateur}],
                "options": {"temperature": 0.3},
            }, timeout=self.timeout)
            r.raise_for_status()
            return (r.json().get("message") or {}).get("content", "").strip()
        except Exception as e:
            return self._reponse_degradee(question, contexte_web, formule, e)

    # -- prompt structure ---------------------------------------------------

    @staticmethod
    def _construire_prompt(question, contexte_web, formule) -> str:
        sections = [f"QUESTION : {question}"]
        # auto-amelioration : injecter les corrections
        ctx_corr = autoamelioration.construire_contexte_corrections(question)
        if ctx_corr:
            sections.append(f"\nCORRECTIONS ANTERIEURES (a ne pas reproduire) :\n{ctx_corr}")
        if contexte_web:
            sections.append(f"\nCONTEXTE WEB (faits verifies) :\n{contexte_web}")
        else:
            sections.append("\nCONTEXTE WEB : (aucun fait externe requis)")
        if formule:
            sections.append(f"\nFORMULE EXACTE (erreur ~0) :\nY = {formule}")
        else:
            sections.append("\nFORMULE EXACTE : (aucune decouverte mathematique)")
        return "\n".join(sections)

    def _reponse_degradee(self, question, contexte_web, formule, erreur) -> str:
        lignes = [f"Question : {question}", ""]
        if contexte_web:
            lignes += ["Faits trouves :", contexte_web, ""]
        if formule:
            lignes += [f"Loi mathematique : Y = {formule}",
                       f"Erreur : {self.derniere_erreur:.3g}", ""]
        lignes.append(f"(LLM indisponible : reponse structuree)")
        return "\n".join(lignes)

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

        return self._generer_cerveau(cerveau, question, contexte_web, formule)

    def executer_detaille(self, question: str, X=None, y=None) -> dict:
        analyse = self.analyser(question)
        experts = analyse["experts"]
        contexte_web = memoire_web.chercher(question) if "web" in experts else ""
        formule = self.resoudre_numerique(X, y) if "math" in experts and X and y else ""
        cerveau = self._choisir_cerveau(experts)
        reponse = self._generer_cerveau(cerveau, question, contexte_web, formule)
        return {"question": question, "experts": experts, "analyse": analyse,
                "cerveau_choisi": cerveau, "contexte_web": contexte_web,
                "formule": formule, "erreur_pgs": self.derniere_erreur,
                "reponse": reponse}
