"""Verification web des reponses (pattern CRITIC) et requete web. Mixin herite par RontoGPT (aura/orchestrateur.py)."""
import logging
import os
from . import memoire_web
from . import llama_cerveau
from . import apis_publiques
from .regles_reponse import _SYSTEME_REQUETE
from .detection_demande import _MOTS_CONTEXTUELS_Q, _VERIF_LONGUEUR_REPONSE, _VERIF_MOTS_FACTUELS

LOG = logging.getLogger("aura.orchestrateur")


class VerificationMixin:
    """Verification web de RontoGPT (methodes de RontoGPT)."""

    @staticmethod
    def _a_besoin_verification(question: str, contexte_web: str, reponse: str) -> bool:
        """Vrai si la reponse merite une preuve web avant livraison."""
        if os.environ.get("AURA_VERIF_WEB") == "0":
            return False
        if contexte_web:
            return False            # deja gelee par les faits web
        if len(reponse) >= _VERIF_LONGUEUR_REPONSE:
            # TEXTE LONG : ce n'est plus une reponse factuelle courte mais un
            # developpement redige, qu'on ne peut pas confronter ligne a
            # ligne a un extrait web.
            return False
        if any(m in question.lower() for m in _MOTS_CONTEXTUELS_Q):
            return False            # question contextuelle : rien a verifier en ligne
        return bool(_VERIF_MOTS_FACTUELS.search(question.lower()))

    @staticmethod
    def _verifier_au_web(question: str, reponse: str) -> str:
        """Re-ancre une reponse factuelle courte sur la preuve web (CRITIC).

        La seconde requete cible la reponse candidate si la premiere ne
        ramene pas de contexte lexicalement pertinent. Le score ne valide
        jamais la veracite : seule la revision ancree sur les sources est
        retenue. Echec web = reponse initiale conservee.
        """
        try:
            ctx = memoire_web.chercher_iteratif(
                question, cible=reponse, max_resultats=2, timeout=6, tours=2)
        except Exception as e:
            LOG.info("[critic] verification impossible (%s) -> reponse conservee", e)
            return reponse
        if not ctx:
            return reponse
        try:
            revisee = llama_cerveau.generer(question, contexte_web=ctx,
                                            max_tokens=150)
        except Exception as e:
            LOG.info("[critic] revision impossible (%s) -> reponse conservee", e)
            return reponse
        if not revisee or revisee.startswith("[RontoGPT]"):
            return reponse
        if revisee.strip() != reponse.strip():
            LOG.info("[critic] reponse revisee apres preuve web")
        return memoire_web.ajouter_citations(revisee, ctx)

    # -- fan-out parallele des experts (idea JEV #1) ------------------------

    def _requete_web(self, question: str) -> str:
        """Mots-cles pour le moteur de recherche (le 1B reformule la question).

        Generation INTERNE : silencieuse (rien n'atteint l'utilisateur). Repli
        sur la question brute si desactivee, trop courte, vide ou en erreur.
        """
        # OPT-IN (AURA_REFORMULATION=1) : une generation 1B SERIELLE avant chaque
        # recherche ajoute de la latence ; son gain de qualite n'est pas mesure.
        if os.environ.get("AURA_REFORMULATION", "0") != "1":
            return question
        if len(question.split()) <= 3:
            return question
        # Une API publique (meteo, devises...) repond deja : pas de 1B en plus.
        if apis_publiques.detecte(question):
            return question
        jeton = llama_cerveau._SILENCIEUX.set(True)
        try:
            brut = llama_cerveau.generer(
                question, max_tokens=24, historique=[],
                systeme=_SYSTEME_REQUETE)
        except Exception as e:                   # noqa: BLE001
            LOG.info("[reformulation] echec : %s", e)
            return question
        finally:
            llama_cerveau._SILENCIEUX.reset(jeton)
        ligne = (brut or "").strip().split("\n")[0].strip(" \"'«»")
        if (not ligne or ligne.startswith("[RontoGPT]") or len(ligne) > 120
                or len(ligne.split()) < 2):
            return question
        LOG.info("[reformulation] %r -> %r", question, ligne)
        return ligne
