"""Orchestrateur autonome : AUCUNE dependance externe (pas Ollama, pas API).

Cerveau unique :
- Llama 3.2 1B Instruct (llama.cpp) : instruction-tuned, bon FR + EN,
  flash attention + KV cache q8_0 + memoire de conversation multi-tours

Plus :
- PGS (gplearn) : formules mathematiques exactes, erreur 0
- DuckDuckGo : faits en temps reel
- Auto-amelioration : corrections injectees dans le prompt

Le routeur (TF-IDF + LogReg) choisit les experts a activer ; Llama synthetise.

Cycle de decision inspire de JEV ultrafast (browser-use) :
- un seul passage : niveau 0 + routeur en parallele (max des durees, pas somme)
- validation de chaque expert AVANT execution (pas de PGS sans donnees)
- fan-out parallele : les experts partent ensemble, echecs isoles
- verification outillee (pattern CRITIC) : le factuel LLM est prouve au web
- confiance calibree (style RouteLLM) : classifieur peu sur -> chemin sur
"""
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from . import expert_symbolique, memoire_web, autoamelioration, filtre_instantane
from . import llama_cerveau
from .routeur import RouteurIntelligent

LOG = logging.getLogger("aura.orchestrateur")

# Verification outillee (pattern CRITIC) : les faits web sont rares chers
# (0,3-1 s), le LLM seul derape parfois (« Sydney » au lieu de Canberra).
# On ne paie la verification que si elle peut CHANGER la reponse :
# - reponse courte (< 200 car) du chemin general (jamais apres un web)
# - question court-factuel (qui/ou/quand/combien + < 9 mots) : le genre de
#   question ou le 1B hallucine. (riches : StyleGuard garantit deja
#   l'ancrage ; contextuelles : rien a verifier en ligne)
# AURA_VERIF_WEB=0 pour desactiver la verification.
_VERIF_MOTS_FACTUELS = re.compile(
    r"\b(qui est|qui a|quelle est|quel est|quand|ou se trouve|combien|"
    r"capitale|population|president|prix de|record)\b")
_VERIF_LONGUEUR_REPONSE = 200
_MOTS_CONTEXTUELS_Q = ("mon ", "ma ", "mes ", "je ", "j'", "tu ", "ton ",
                       "votre ", "prenom", "nom")

# Seuil de confiance (style RouteLLM) : sous ce score max du classifieur,
# on ne fait pas confiance a son choix -> chemin sur 'general' (le LLM
# tranche avec tout son contexte). Calibre sur 14 questions reelles :
# cas clairs 0.50-0.85, ambigus 0.47-0.55 -> 0.50 ne demote que l'incertain
# reel ; les maths timides sont de toute facon rattrapees par le niveau 0.
# AURA_SEUIL_CONFIANCE=0.45 pour desserrer, 0.60 pour durcir.
_SEUIL_CONFIANCE = float(os.environ.get("AURA_SEUIL_CONFIANCE", "0.50"))


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

    # -- decision en un cycle (idea JEV #3) ---------------------------------

    def _decider(self, question: str):
        """Un seul cycle de decision : niveau 0 + routeur en parallele.

        JEV n'observe qu'une fois par cycle de decision ; ici pareil : le
        filtre instantane (cache + maths directes) et le routage partent
        ensemble, la decision complete coute max(durees) au lieu de la
        somme. Sur un hit niveau 0, le routeur est abandone sans attendre.
        Renvoie (instant, analyse) — analyse vaut None si le niveau 0 a
        tranche tout seul.
        """
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            f_instant = pool.submit(filtre_instantane.repondre, question)
            f_route = pool.submit(self.analyser, question)
            instant = f_instant.result()
            if instant is not None:
                f_route.cancel()   # parti en parallele : il finit seul, on l'ignore
                return instant, None
            return instant, f_route.result()
        finally:
            pool.shutdown(wait=False)

    # -- validation avant execution (idea JEV #2) ---------------------------

    @staticmethod
    def _donnees_valides(X, y) -> bool:
        """Donnees numeriques exploitables par le PGS (>= 2 points alignes)."""
        if X is None or y is None:
            return False
        try:
            return len(X) == len(y) and len(X) >= 2
        except TypeError:
            return False

    def _valider_experts(self, analyse, X=None, y=None) -> set:
        """Garde-fou : verifier la confiance PUIS chaque expert AVANT exécution.

        JEV verifie chaque cible avant le clic ; RouteLLM route selon la
        qualite predite. Ici, deux controles :
        1. confiance : le score max du classifieur sous _SEUIL_CONFIANCE ->
           son choix n'est pas fiable, on prend le chemin sur ('general') ;
        2. PGS sans donnees numeriques exploitables -> ignore (calcul
           genetique lance pour rien). Un ensemble vide retombe sur 'general'.
        """
        experts = set(analyse.get("experts") or ())
        scores = analyse.get("scores") or {}
        if scores:
            meilleure = max(scores.values())
            if meilleure < _SEUIL_CONFIANCE:
                LOG.info("[routeur] confiance %.2f < %.2f -> chemin sur 'general'",
                         meilleure, _SEUIL_CONFIANCE)
                return {"general"}
        if "math" in experts and not self._donnees_valides(X, y):
            LOG.info("[garde-fou] 'math' route sans donnees numeriques -> ignore")
            experts.discard("math")
            if not experts:
                experts.add("general")
        return experts

    # -- verification outillee (pattern CRITIC) -----------------------------

    @staticmethod
    def _a_besoin_verification(question: str, contexte_web: str, reponse: str) -> bool:
        """Vrai si la reponse merite une preuve web avant livraison."""
        if os.environ.get("AURA_VERIF_WEB") == "0":
            return False
        if contexte_web:
            return False            # deja gelee par les faits web
        if len(reponse) >= _VERIF_LONGUEUR_REPONSE:
            return False            # mode riche : redaction guidee, pas verifiable ligne a ligne
        if any(m in question.lower() for m in _MOTS_CONTEXTUELS_Q):
            return False            # question contextuelle : rien a verifier en ligne
        return bool(_VERIF_MOTS_FACTUELS.search(question.lower()))

    @staticmethod
    def _verifier_au_web(question: str, reponse: str) -> str:
        """Re-ancre une reponse factuelle courte sur la preuve web (CRITIC).

        Pas d'heuristique fragile de comparaison de mots (les extraits citent
        souvent la reponse fausse comme « confusion connue ») : le modele
        REpond une seconde fois, ancre sur les faits trouves. Une seule
        passe — le resultat n'est pas re-verifie (pas de boucle).
        Echec web = reponse initiale conservee (jamais bloquant).
        """
        try:
            ctx = memoire_web.chercher(question, max_resultats=2, timeout=6)
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
        if not revisee or revisee.startswith("[Aura]"):
            return reponse
        if revisee.strip() != reponse.strip():
            LOG.info("[critic] reponse revisee apres preuve web")
        return revisee

    # -- fan-out parallele des experts (idea JEV #1) ------------------------

    def _executer_experts(self, experts, question, X, y, riche):
        """Lance les experts en parallele (web ∥ PGS), un aller-retour.

        Deux experts -> deux threads (le reseau et le CPU se recouvrent) ;
        un seul expert -> appel direct, zero surcout. Un expert qui echoue
        ne bloque jamais l'autre : chaque tache est isolee, echec = chaine
        vide. Renvoie (contexte_web, formule).
        """
        taches = {}
        if "web" in experts:
            fn = memoire_web.chercher_enrichi if riche else memoire_web.chercher
            taches["web"] = (fn, (question,))
        if "math" in experts:
            taches["pgs"] = (self.resoudre_numerique, (X, y))

        if not taches:
            return "", ""

        def _isole(nom, fn, args):
            try:
                return fn(*args)
            except Exception as e:          # un expert en echec = chaine vide
                LOG.warning("[expert %s] echec : %s", nom, e)
                return ""

        resultats = {}
        if len(taches) == 1:
            nom, (fn, args) = next(iter(taches.items()))
            resultats[nom] = _isole(nom, fn, args)
        else:
            # threads explicites : un par expert, paralleisation GARANTIE
            # (un ThreadPoolExecutor peut reutiliser un worker inactif et
            # executer les deux taches en serie quand elles sont rapides)
            threads = []
            for nom, (fn, args) in taches.items():
                def _courir(n=nom, f=fn, a=args):
                    resultats[n] = _isole(n, f, a)
                t = threading.Thread(target=_courir, name=f"aura-{nom}",
                                     daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
        return resultats.get("web", ""), resultats.get("pgs", "")

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
        # ── CYCLE UNIQUE (idea JEV #3) ─────────────────────────────────
        # maths directes + cache semantique + routage en UN passage : la
        # majorite des questions quotidiennes n'ont PAS besoin du LLM.
        instant, analyse = self._decider(question)
        if instant is not None and not (X and y):
            return {"question": question, "experts": {"instantane"},
                    "analyse": {"methodes": ["niveau0"]},
                    "cerveau_choisi": "niveau0-instantane",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": instant}

        # ── NIVEAUX 1-2 : experts + LLM ───────────────────────────────
        # VALIDATION AVANT EXECUTION (idea JEV #2 + confiance RouteLLM) :
        # chaque expert est verifie avant d'etre lance.
        experts = self._valider_experts(analyse, X, y)
        riche = self._est_complexe(question)
        # FAN-OUT PARALLELE (idea JEV #1) : web ∥ PGS en un aller-retour
        contexte_web, formule = self._executer_experts(
            experts, question, X, y, riche)
        reponse = self._generer(question, contexte_web, formule, riche=riche)
        # VERIFICATION OUTILLEE (pattern CRITIC) : preuve web avant livraison
        if self._a_besoin_verification(question, contexte_web, reponse):
            reponse = self._verifier_au_web(question, reponse)
            # RECONSOLIDATION : la reponse verifiee remplace l'ancienne au
            # cache (sinon une reponse fausse d'avant reste collée a vie)
            filtre_instantane.mettre_a_jour(question, reponse)
        # memorise pour les futures questions (cache semantique + conversation)
        filtre_instantane.enregistrer(question, reponse)
        self._historique.append({"role": "user", "content": question})
        self._historique.append({"role": "assistant", "content": reponse})
        return {"question": question, "experts": experts, "analyse": analyse,
                "cerveau_choisi": "llama-3.2-1b", "contexte_web": contexte_web,
                "formule": formule, "erreur_pgs": self.derniere_erreur,
                "reponse": reponse}
