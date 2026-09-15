"""Orchestrateur : comprend, route, redige.

Deux cerveaux interchangeables (meme interface `generer()`) :
- "ollama" (defaut) : n'importe quel LLM local OpenAI-compatible, fiable
- "mamba" : vrai State Space Model lineaire (memoire a taille fixe, rapide CPU) —
  repli automatique sur Ollama si torch/transformers ou le modele manquent
"""
import logging

import requests

from . import expert_symbolique, memoire_web

LOG = logging.getLogger("aura.orchestrateur")

PROMPT_SYSTEME = (
    "Tu es Aura, une IA hybride neuro-symbolique. On te donne une question "
    "utilisateur, un CONTEXTE WEB (faits recents, peut etre vide) et une "
    "FORMULE SYMBOLIQUE decouverte par un moteur mathematique exact (peut etre "
    "vide). Reponds en francais, en 5 phrases maximum. Cite les faits du "
    "contexte web si presents. Si une formule est fournie, explique-la "
    "simplement. Ne jamais inventer de faits : si l'information manque, dis-le."
)


class Aura1B:
    """Chef d'orchestre des trois piliers."""

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

    # -- pilier langage ----------------------------------------------------

    def generer(self, question: str, contexte_web: str, formule: str) -> str:
        """Redaction finale : Mamba si demande et dispo, sinon Ollama, sinon template."""
        if self._mamba_actif:
            try:
                if self._mamba is None:
                    from .mamba_orchestrateur import OrchestrateurMamba
                    self._mamba = OrchestrateurMamba(grande=self._mamba_grande)
                return self._mamba.generer(question, contexte_web, formule)
            except Exception as e:  # torch absent, modele inaccessible, OOM...
                LOG.warning("Mamba indisponible (%s) : repli sur Ollama", e)
                self._mamba_actif = False
        return self._generer_ollama(question, contexte_web, formule)

    def _generer_ollama(self, question: str, contexte_web: str, formule: str) -> str:
        """Redaction finale par le modele de langage local (fallback template si absent)."""
        utilisateur = (
            f"QUESTION : {question}\n\nCONTEXTE WEB :\n{contexte_web or '(aucun)'}\n\n"
            f"FORMULE SYMBOLIQUE EXACTE :\n{formule or '(aucune)'}"
        )
        try:
            r = requests.post(f"{self.hote}/api/chat", json={
                "model": self.modele, "stream": False,
                "messages": [{"role": "system", "content": PROMPT_SYSTEME},
                             {"role": "user", "content": utilisateur}],
                "options": {"temperature": 0.3},
            }, timeout=self.timeout)
            r.raise_for_status()
            return (r.json().get("message") or {}).get("content", "").strip()
        except Exception as e:
            return self._reponse_degradée(question, contexte_web, formule, e)

    def _reponse_degradée(self, question, contexte_web, formule, erreur) -> str:
        """Sans LLM : synthese structuree honnete (jamais de faux texte 'genere')."""
        lignes = [f"Question : {question}", ""]
        if contexte_web:
            lignes += ["Faits trouves sur le web :", contexte_web, ""]
        if formule:
            lignes += [f"Loi mathematique decouverte (exacte) : Y = {formule}",
                       f"Erreur quadratique moyenne : {self.derniere_erreur:.3g}", ""]
        lignes.append(f"(LLM local indisponible ({erreur}) : reponse sans redaction.)")
        return "\n".join(lignes)

    # -- pilier logique ----------------------------------------------------

    def resoudre_numerique(self, X, y) -> str:
        """Decouvre la loi Y = f(X) ; renvoie la formule (vide si impossible)."""
        formule = self.expert.resoudre(X, y)
        self.derniere_formule = formule
        self.derniere_erreur = self.expert.erreur
        return formule

    # -- pipeline complet ----------------------------------------------------

    def executer(self, question: str, X=None, y=None) -> str:
        """Pipeline : routage -> web -> PGS -> redaction."""
        contexte_web = memoire_web.chercher(question) if memoire_web.a_besoin_web(question) else ""
        formule = self.resoudre_numerique(X, y) if X is not None and y is not None else ""
        return self.generer(question, contexte_web, formule)

    def executer_detaille(self, question: str, X=None, y=None) -> dict:
        """Variante qui expose les etapes intermediaires (pour le CLI et les tests)."""
        web_actif = memoire_web.a_besoin_web(question)
        contexte_web = memoire_web.chercher(question) if web_actif else ""
        formule = self.resoudre_numerique(X, y) if X is not None and y is not None else ""
        reponse = self.generer(question, contexte_web, formule)
        return {"question": question, "web_actif": web_actif, "contexte_web": contexte_web,
                "formule": formule, "erreur_pgs": self.derniere_erreur, "reponse": reponse}
