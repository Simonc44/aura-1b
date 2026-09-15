"""Orchestrateur : comprend, route, redige.

Deux cerveaux interchangeables (meme interface `generer()`) :
- "ollama" (defaut) : n'importe quel LLM local OpenAI-compatible
- "mamba" : vrai State Space Model lineaire (130M, CPU)
Routeur intelligent : TF-IDF + LogReg (150 exemples) + cosinus prototypes.
Synthese structuree : le LLM recoit des sections CLAIRES (pas un blob).
"""
import logging

import requests

from . import expert_symbolique, memoire_web
from .routeur import RouteurIntelligent

LOG = logging.getLogger("aura.orchestrateur")

# prompt de synthese : structure claire pour que le LLM fusionne proprement
_PROMPT_SYSTEME = (
    "Tu es Aura, une IA neuro-symbolique. Tu recois 3 sections distincies :\n"
    "- CONTEXTE WEB : faits verifies recuperes sur internet\n"
    "- FORMULE EXACTE : une loi mathematique precise (decouverte par un moteur "
    "evolutionnaire, erreur proche de 0)\n"
    "- QUESTION : ce que l'utilisateur demande\n\n"
    "CONSIGNES :\n"
    "1. Si une formule est presente, explique-la simplement en francais.\n"
    "2. Si le contexte web contient des faits, integre-les dans ta reponse.\n"
    "3. Ne JAMAIS inventer de faits non fournis.\n"
    "4. Reponds en 3 phrases maximum, claires et directes."
)


class Aura1B:
    """Chef d'orchestre des trois piliers + routeur intelligent."""

    def __init__(self, hote="http://localhost:11434", modele="qwen2.5:1.5b-instruct",
                 timeout=90, cerveau="ollama", mamba_grande=False):
        self.hote = hote.rstrip("/")
        self.modele = modele
        self.timeout = timeout
        self.expert = expert_symbolique.ExpertSymbolique()
        self.derniere_formule = ""
        self.derniere_erreur = None
        self._mamba = None
        self._mamba_actif = (cerveau == "mamba")
        self._mamba_grande = mamba_grande
        self._routeur = RouteurIntelligent()

    # -- routage intelligent ------------------------------------------------

    def analyser(self, question: str) -> dict:
        """Analyse la question et decide quels experts activer."""
        try:
            return self._routeur.classer(question)
        except Exception as e:
            LOG.warning("[routeur] echec LogReg (%s), fallback mots-cles", e)
            return self._routeur.routeur_classique(question)

    # -- pilier langage ----------------------------------------------------

    def generer(self, question: str, contexte_web: str, formule: str) -> str:
        """Redaction finale : Mamba si demande, sinon Ollama, sinon template."""
        if self._mamba_actif:
            if self._mamba is None:
                from .mamba_orchestrateur import OrchestrateurMamba
                try:
                    self._mamba = OrchestrateurMamba(grande=self._mamba_grande)
                except Exception as e:
                    LOG.warning("Mamba init impossible (%s), repli Ollama", e)
                    self._mamba_actif = False
            if self._mamba is not None:
                try:
                    return self._mamba.generer(question, contexte_web, formule)
                except Exception as e:
                    LOG.warning("Mamba indisponible (%s), repli Ollama", e)
                    self._mamba_actif = False
        return self._generer_ollama(question, contexte_web, formule)

    def _generer_ollama(self, question, contexte_web, formule) -> str:
        """LLM local via Ollama (fallback template si indisponible)."""
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

    @staticmethod
    def _construire_prompt(question, contexte_web, formule) -> str:
        """Construit un prompt structure avec 3 sections claires."""
        sections = [f"QUESTION : {question}"]
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
        """Template honnête si aucun LLM disponible."""
        lignes = [f"Question : {question}", ""]
        if contexte_web:
            lignes += ["Faits trouves sur le web :", contexte_web, ""]
        if formule:
            lignes += [f"Loi mathematique exacte : Y = {formule}",
                       f"Erreur : {self.derniere_erreur:.3g}", ""]
        lignes.append(f"(LLM indisponible ({erreur}) : reponse structuree sans redaction.)")
        return "\n".join(lignes)

    # -- pilier logique ----------------------------------------------------

    def resoudre_numerique(self, X, y) -> str:
        """Decouvre la loi Y = f(X) ; renvoie la formule (vide si impossible)."""
        formule = self.expert.resoudre(X, y)
        self.derniere_formule = formule
        self.derniere_erreur = self.expert.erreur
        return formule

    # -- pipeline complet ---------------------------------------------------

    def executer(self, question: str, X=None, y=None) -> str:
        """Pipeline complet : routage intelligent -> experts -> synthese."""
        analyse = self.analyser(question)
        experts = analyse["experts"]

        contexte_web = ""
        if "web" in experts:
            contexte_web = memoire_web.chercher(question)

        formule = ""
        if "math" in experts and X is not None and y is not None:
            formule = self.resoudre_numerique(X, y)

        return self.generer(question, contexte_web, formule)

    def executer_detaille(self, question: str, X=None, y=None) -> dict:
        """Variante qui expose les etapes intermediaires (CLI + tests)."""
        analyse = self.analyser(question)
        experts = analyse["experts"]
        contexte_web = memoire_web.chercher(question) if "web" in experts else ""
        formule = self.resoudre_numerique(X, y) if "math" in experts and X is not None and y is not None else ""
        reponse = self.generer(question, contexte_web, formule)
        return {"question": question, "experts": experts, "analyse": analyse,
                "contexte_web": contexte_web, "formule": formule,
                "erreur_pgs": self.derniere_erreur, "reponse": reponse}
