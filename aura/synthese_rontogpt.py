"""Synthese de la reponse de RontoGPT : prompt systeme, garde-fous de
contamination et generation. Mixin herite par RontoGPT (aura/orchestrateur.py)."""
import logging
from typing import TYPE_CHECKING, Callable

from . import autoamelioration
from . import agents
from .regles_reponse import _rejoue_le_tour_precedent, _reste_substantiel, _sans_amorce_meta
from .detection_demande import _RE_SORTIE_LONGUE, _SYSTEME_RAISONNEMENT, _SYSTEME_REDACTION, _est_demande_ecriture, _est_redaction, _sans_accents

if TYPE_CHECKING:
    from .memoire_conversation import MemoireConversation

LOG = logging.getLogger("aura.orchestrateur")


class SyntheseMixin:
    """Generation de la reponse finale (methodes de RontoGPT)."""

    # CONTRAT STATIQUE (jamais execute) : ces noms sont portes par RontoGPT
    # (aura/orchestrateur.py) au montage des mixins. Les declarer ici, sous
    # `TYPE_CHECKING`, rend le mixin verifiable par mypy sans import
    # circulaire ni dependance au runtime : le bloc disparait a l'execution.
    if TYPE_CHECKING:
        _llama: Callable[..., str] | None
        _historique: "MemoireConversation"
        _special_traite: bool

    def _generer(self, question: str, contexte_web: str, formule: str,
                 contexte_faits: str = "",
                 personnalite: str | None = None,
                 categorie: str = "",
                 est_def: bool = False,
                 est_def_lookup: bool = False,
                 max_tokens: int | None = None,
                 think: bool | None = None) -> str:
        if self._llama is None:
            from .llama_cerveau import generer as llama_generer
            self._llama = llama_generer
        # MIMETISME (few-shot) : une brique d'exemples « calibre grand
        # modele » guide le style — dans le PROMPT SYSTEME, jamais collee
        # a la question : le 1B recopiait les exemples (« Exemples de
        # style attendus... ») au lieu de repondre a la question.
        systeme_final = personnalite
        # DEMANDE DE DEVELOPPEMENT (redaction longue) : meme remede que
        # l'EXCEPTION AUTO-THINK juste en dessous, autre cause. Le prompt
        # complet porte des CONSIGNES D'OUTIL et des briques de style
        # COPIABLES, et sur une demande de redaction le 1B les RECOPIE au
        # lieu de rediger (mesure du banc, cf. _SYSTEME_REDACTION). On ne
        # garde donc AUCUNE brique copiable : ni exemples de style, ni
        # lexique de concepts, ni consignes d'outil.
        redaction = bool(not think and (
            _est_redaction(question)
            or _est_demande_ecriture(question)
            or _RE_SORTIE_LONGUE.search(_sans_accents(question))))
        # EXCEPTION AUTO-THINK : sur un puzzle/enigme (think=True), la
        # CORRECTION prime sur le STYLE. La brique d'exemples est du
        # contexte COPIABLE et le 1B la recopie au lieu de raisonner —
        # bug reel (10 oct. 2026) : l'enigme « escargot » rendait
        # l'exemple du ciel bleu, ou la question precedente. On ne
        # l'injecte donc PAS quand la question demande a raisonner.
        if not think and not redaction:
            try:
                brique = agents.exemple_style(question, categorie)
                if brique:
                    systeme_final = (f"{personnalite}\n\n{brique}"
                                     if personnalite else brique)
            except Exception as e:
                # Exemple de style absent : la reponse est moins bien guidee,
                # jamais fausse. Debug suffit — c'est une option de style.
                LOG.debug("[synthese] exemple de style ignore : %s", e)
        # DETECTEUR DE CONCEPTS : la question releve-t-elle d'un concept
        # abstrait connu (bibliotheque MiniLM) ? Alors un bloc LEXIQUE est
        # ajoute au prompt systeme : le 1B recopie une formulation exacte
        # et calibree au lieu d'improviser sur un mot qu'il ne maitrise
        # pas. Desactive par AURA_CONCEPTS=0.
        # Idem pour l'ancrage concepts : du texte a recopier detourne un
        # puzzle. Reserve aux questions de restitution (pas think).
        if not think and not redaction:
            try:
                from . import concepts
                ancrage = concepts.ancrage(question)
                if ancrage:
                    systeme_final = (f"{systeme_final}\n{ancrage}"
                                     if systeme_final else ancrage)
            except Exception:  # noqa: BLE001
                pass
        # OUTIL RLM (MIT) : la question demande-t-elle l'historique ?
        # Resolution symbolique (recherche fichier arriere) — zero LLM,
        # zero token d'historique injecte.
        resultat_rlm = self._historique.tourner(question)
        if resultat_rlm is not None:
            return self._llama(
                f"{question}\n\nRESULTAT DE LA RECHERCHE DANS L'HISTORIQUE :\n"
                f"{resultat_rlm}", contexte_web, formule,
                historique=self._historique,
                contexte_faits=contexte_faits,
                systeme=systeme_final,
                categorie=categorie,
                max_tokens=max_tokens)
        # OUTIL LOCAL : « lis le fichier <chemin> » -> contenu injecte
        # (resolution symbolique, meme esprit que le RLM — zero LLM).
        contenu_fichier = self._historique.lire_fichier(question)
        if contenu_fichier is not None:
            return self._llama(
                f"{question}\n\nCONTENU DU FICHIER :\n{contenu_fichier}",
                contexte_web, formule,
                historique=self._historique,
                contexte_faits=contexte_faits,
                systeme=systeme_final,
                categorie=categorie,
                max_tokens=max_tokens)
        # FILET FINAL (anti-hallucination) : question de LOOKUP (definition,
        # fait chiffre) SANS aucun ancrage (web injoignable, graphe muet, pas
        # d'expert special) -> on n'invente pas : garde-fou honnete + contexte
        # du graphe s'il existe. Un « je ne peux pas verifier » vaut mieux
        # qu'une definition fausse dite avec assurance.
        # `est_def_lookup` et non `est_def` : une demande de DEVELOPPEMENT
        # (« explique la photosynthese ») n'est pas une definition a verifier,
        # cf. _RE_DEF_LOOKUP. Les deux sont calcules sur la question
        # ORIGINALE (jamais sur la question + brique few-shot, cf.
        # executer_detaille).
        if (est_def_lookup and not contexte_web and not contexte_faits
                and not self._special_traite):
            return ("Je ne peux pas verifier cette definition en ce moment "
                    "(hors-ligne et absente de ma base de faits). Pour ne "
                    "pas t'inventer une reponse fausse, je prefere m'abstenir. "
                    "Reessaie avec internet, ou apprends-moi ce fait : "
                    "graphe_faits.ajouter(...).")
        # POUR UNE QUESTION DE RAISONNEMENT : prompt systeme MINIMAL,
        # sans les consignes d'outil que le 1B recopie (cf. la constante).
        systeme_envoye = (
            _SYSTEME_RAISONNEMENT if think
            else _SYSTEME_REDACTION if redaction
            else self._historique.bloc_systeme(systeme_final))
        # Le brouillon de raisonnement n'herite pas de l'historique distant :
        # garder au plus le dernier echange utile pour resoudre les references
        # conversationnelles, sans injecter les vieux exemples/corrections.
        historique_generation = (
            self._historique.fenetre()[-2:] if think else self._historique)
        reponse = self._llama(question, contexte_web, formule,
                              historique=historique_generation,
                              contexte_faits=contexte_faits,
                              systeme=systeme_envoye,
                              categorie=categorie,
                              max_tokens=max_tokens,
                              **({"think": think} if think else {}))
        # -- GARDE-FOU CONTAMINATION (cf. _RE_AMORCE_META) ---------------
        # On retire l'amorce meta ; si la reponse n'etait que cette amorce,
        # ou si elle rejoue le tour precedent, on regenere SANS HISTORIQUE.
        reponse = _sans_amorce_meta(reponse)
        if (not _reste_substantiel(reponse)
                or _rejoue_le_tour_precedent(reponse, self._historique)):
            LOG.info("[anti-contamination] tour rejoue ou amorce meta vide "
                     "-> regeneration sans historique")
            propre = _sans_amorce_meta(
                self._llama(question, contexte_web, formule,
                            historique=[],
                            contexte_faits=contexte_faits,
                            systeme=systeme_envoye,
                            categorie=categorie,
                            max_tokens=max_tokens,
                            **({"think": think} if think else {})))
            if propre.strip():
                reponse = propre
        return reponse

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

