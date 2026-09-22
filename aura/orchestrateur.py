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
from . import llama_cerveau, raisonneur, graphe_faits
from . import logique as solveur_logique
from . import potcode
from . import agents
from . import adaptateurs
from . import serveur_lora
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

# experts speciaux : signatures lexicales conservatrices (faux positif =
# juste un chemin normal, jamais une reponse fausse)
_ENIGME_LOGIQUE = re.compile(
    r"\b(chevaliers?|menteus\w*|mentent|mensonge|enigme de logique|"
    r"puzzle de logique|devinette de logique|dit la verite|"
    r"dit toujours vrai|dit toujours la verite)\b")
_DEMANDE_CODE = re.compile(
    r"\b(ecris|ecrire|implemente|genere|developpe|code)\b[^.?!]{0,60}"
    r"\b(fonction|python|code|script|programme|algorithme)\b")

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
                 riche: bool = False, contexte_faits: str = "",
                 personnalite: str | None = None,
                 categorie: str = "", complexe: bool = False) -> str:
        if self._llama is None:
            from .llama_cerveau import generer as llama_generer
            self._llama = llama_generer
        # AURA_SERVEUR=1 : le Dynamic Compute (reflection masquee) remplace
        # le multi-pass in-process (qui chargerait le cerveau a double).
        if riche and serveur_lora._actifs():
            return self._llama(question, contexte_web, formule,
                               historique=self._historique,
                               contexte_faits=contexte_faits,
                               systeme=personnalite,
                               categorie=categorie,
                               complexe=True)
        if riche:
            from .llama_cerveau import generer_riche
            return generer_riche(question, contexte_web, formule,
                                 historique=self._historique)
        return self._llama(question, contexte_web, formule,
                           historique=self._historique,
                           contexte_faits=contexte_faits,
                           systeme=personnalite,
                           categorie=categorie,
                           complexe=complexe)

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

    # -- raisonnement Program-of-Thoughts -----------------------------------

    @staticmethod
    def _est_puzzle(question: str) -> bool:
        """Puzzle d'ages/relations : le terrain ou PoT rapporte le plus."""
        try:
            return raisonneur.est_puzzle(question)
        except Exception:
            return False

    # -- enigme logique (mini-SAT pur Python) -------------------------------

    def _resoudre_par_logique(self, question: str) -> str | None:
        """Le 1B formalise (entites/domaine/dits), le solveur deduit exactement.

        Avec UNE relance de correction : si la formalisation est rejetee,
        le modele voit SON texte + le motif du refus (self-debug, une seule
        fois). UNIQUE -> reponse garantie ; MULTIPLE/AUCUNE -> None.
        """
        try:
            # cadrage « Enigme : … / Formalisme : » : mesuré en réel, c'est
            # celui qui fait respecter le format au 1B (noms exacts)
            brut = llama_cerveau.generer(
                f"Enigme :\n{question}\n\nFormalisme :",
                systeme=solveur_logique._SYSTEME_LOGIQUE,
                max_tokens=300)
            try:
                puzzle = solveur_logique.parser_puzzle(brut)
            except ValueError as e:
                LOG.info("[logique] formalisation rejetee (%s) -> 1 relance", e)
                second = llama_cerveau.generer(
                    question + "\n\nTa reponse precedente etait :\n" + brut
                    + "\n\nElle a ete REJETEE car : " + str(e)
                    + "\nReponds a nouveau, STRICTEMENT selon le format, "
                      "avec les vrais noms de l'enigme et SANS inventer de "
                      "contrainte.",
                    systeme=solveur_logique._SYSTEME_LOGIQUE,
                    max_tokens=300)
                puzzle = solveur_logique.parser_puzzle(second)
            r = solveur_logique.resoudre(puzzle)
            if r["statut"] != "unique":
                LOG.info("[logique] statut %s -> chemin normal", r["statut"])
                return None
            verification = solveur_logique.bloc_verification(
                puzzle, r["solutions"])
            finale = llama_cerveau.generer(question, contexte_web=verification,
                                           max_tokens=150)
            LOG.info("[logique] enigme resolue exactement")
            return finale
        except (ValueError, Exception) as e:      # noqa: B014 — jamais bloquant
            LOG.info("[logique] formalisation/refus (%s) -> chemin normal", e)
            return None

    # -- demande de code (PoT-code) -----------------------------------------

    def _resoudre_par_code(self, question: str, mode_auto: bool = False) -> str | None:
        """Le 1B ecrit fonction + asserts, la sandbox les execute.

        Un assert rate = code refuse = None (jamais de code casse livre).
        mode_auto : l'agent debugger recursif prend le relais (jusqu'a 3
        essais, l'erreur exacte renvoyee au modele a chaque relance) ;
        echec de la boucle = None (chemin normal, jamais pire qu'avant).
        """
        try:
            brut = llama_cerveau.generer(question,
                                         systeme=potcode._SYSTEME_CODE,
                                         max_tokens=400)
            code = potcode.extraire_code(brut)
            if not code:
                return None
            rapport = potcode.verifier_code(code)
            LOG.info("[potcode] valide : %d asserts", rapport["nb_asserts"])
            return "```python\n" + code + "\n```\n\n(Code verifie : " \
                   f"{rapport['nb_asserts']} asserts passes en sandbox.)"
        except (ValueError, Exception) as e:      # noqa: B014 — jamais bloquant
            LOG.info("[potcode] code refuse (%s) -> chemin normal", e)
            if not mode_auto:
                return None
            # AGENT 4 (debugger recursif) : boucle de self-debug max 3
            try:
                return agents.boucle_code(question, max_tentatives=3)
            except Exception as e2:
                LOG.info("[agents] boucle code impossible (%s) -> chemin normal", e2)
                return None

    def _expert_special(self, question: str, X, y) -> dict | None:
        """Routage des experts speciaux, dans l'ordre de specialisation.

        Donnees numeriques -> PGS direct (pas de PoT/logique). Renvoie le
        dict de reponse si un expert a tranché, sinon None.
        """
        if X and y:
            return None
        if _ENIGME_LOGIQUE.search(question.lower()):
            rep = self._resoudre_par_logique(question)
            if rep is not None:
                return self._resultat_special(question, rep, "logique-1b+sat")
        if _DEMANDE_CODE.search(question.lower()):
            rep = self._resoudre_par_code(question, mode_auto=True)
            if rep is not None:
                return self._resultat_special(question, rep, "potcode-1b+sandbox")
        return None

    def _resultat_special(self, question: str, reponse: str, cerveau: str) -> dict:
        filtre_instantane.enregistrer(question, reponse)
        return {"question": question, "experts": {cerveau.split("-")[0]},
                "analyse": {"methodes": ["expert_special"]},
                "cerveau_choisi": cerveau, "contexte_web": "", "formule": "",
                "erreur_pgs": None, "reponse": reponse}

    def _resoudre_par_pot(self, question: str) -> str | None:
        """Passe PoT : le 1B ecrit les etapes, l'AST verifie chaque calcul.

        Renvoie None si le modele ne produit pas >= 2 etapes calculables —
        dans ce cas on retombe sur le chemin normal (jamais pire qu'avant).
        """
        try:
            brut = llama_cerveau.generer(question, systeme=raisonneur._SYSTEME_POT,
                                         max_tokens=280)
        except Exception as e:
            LOG.info("[pot] generation impossible : %s", e)
            return None
        etapes = raisonneur.extraire_etapes(brut)
        if len(etapes) < 2:
            LOG.info("[pot] pas assez d'etapes calculables (%d) -> chemin normal",
                     len(etapes))
            return None
        try:
            resultats = raisonneur.verifier_calculs(etapes)
        except ValueError as e:
            LOG.info("[pot] calcul illisible (%s) -> chemin normal", e)
            return None
        verification = raisonneur.construire_verification(etapes, resultats)
        try:
            finale = llama_cerveau.generer(question,
                                           contexte_web=verification,
                                           max_tokens=120)
        except Exception as e:
            LOG.info("[pot] synthese impossible : %s", e)
            return None
        LOG.info("[pot] puzzle resolu : %d etapes verifiees", len(etapes))
        return finale

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

        # EXPERTS SPECIAUX (logique exacte, code sandboxe) : signatures
        # conservatrices, repli cascade vers le chemin normal. AVANT le
        # fan-out : une enigme de logique n'a pas besoin de DuckDuckGo.
        special = self._expert_special(question, X, y)
        if special is not None:
            return special

        # FAN-OUT PARALLELE (idea JEV #1) : web ∥ PGS en un aller-retour
        contexte_web, formule = self._executer_experts(
            experts, question, X, y, riche)

        # AGENT 2 (compresseur RAG) : le contexte web brut est filtre —
        # seules les phrases utiles a la question alimentent le LLM
        if contexte_web:
            try:
                contexte_web = agents.compresser(contexte_web, question,
                                                 riche=riche)
            except Exception as e:
                LOG.info("[agents] compression impossible (%s) -> texte brut", e)
            # AGENT 2b (hyper-compression) : phrases -> triplets semantiques
            # (sujet | relation | objet) — protege la fenetre du 1B et
            # nourrit le graphe de faits au passage
            if not riche:
                try:
                    contexte_web = agents.en_triplets(contexte_web, question)
                except Exception as e:
                    LOG.info("[agents] triplets impossible (%s) -> texte compresse", e)

        # AGENT 1 (planificateur) : les taches actionables complexes sont
        # decoupees en 2-3 etapes AVANT la redaction
        plan = ""
        if not riche and not formule:
            try:
                plan = agents.planifier(question) or ""
            except Exception as e:
                LOG.info("[agents] planification impossible (%s) -> sans plan", e)
            if plan:
                LOG.info("[agents] plan insere : %d lignes", len(plan.splitlines()))

        # RAISONNEMENT PoT (Program-of-Thoughts) : les puzzles d'ages/
        # relations sans donnees numeriques vont au raisonneur — le 1B
        # ecrit les etapes, l'AST les verifie exactement. Repli auto.
        # PAS de condition !riche : les puzzles sont structurellement
        # longs (>= 9 mots = mode riche) — leur signature (ages+relations)
        # est plus fiable que le comptage de mots.
        if not (X and y) and not contexte_web and self._est_puzzle(question):
            pot = self._resoudre_par_pot(question)
            if pot is not None:
                filtre_instantane.enregistrer(question, pot)
                return {"question": question, "experts": experts | {"pot"},
                        "analyse": analyse, "cerveau_choisi": "pot-1b+ast",
                        "contexte_web": "", "formule": "",
                        "erreur_pgs": None, "reponse": pot}

        # GRAPHE DE FAITS (MiniRAG-lite) : retrouver au lieu de deviner —
        # uniquement des faits VERIFIES web (jamais d'hallucination dedans).
        contexte_faits = "" if contexte_web else \
            graphe_faits.chercher(question)
        # AGENT 5 (personnalite) : prompt systeme ajuste a la categorie
        # routee ; None = prompt de base du cerveau (comportement d'avant).
        # Le plan (agent 1) complete la question ; web compresse (agent 2)
        # et personnalite sont passes au cerveau en meme temps.
        try:
            personnalite = agents.composer_personnalite(experts, question)
        except Exception:
            personnalite = None
        question_envoyee = f"{question}\n\nPLAN A SUIVRE :\n{plan}" if plan else question
        # HOT-SWAP ADAPTATEUR (LoRA) : si un adaptateur existe pour la
        # categorie routee, il est monte sur le contexte charge SANS
        # recharger le modele (AURA_ADAPTATEURS=1 pour activer ; defaut
        # OFF = zero surcout, chemin inchange). Echec -> cerveau brut.
        categorie = ("web" if "web" in experts else
                     "math" if "math" in experts else
                     "code" if _DEMANDE_CODE.search(question) else "general")
        self._derniere_categorie = categorie
        # MULTI-LORA SIMULTANE : les scores du routeur (probabilites) sont
        # converts en poids d'adaptateurs — une question a deux sujets
        # profite des deux specialites (ex : {"math": 0.7, "web": 0.3}).
        # Defaut : la categorie dominante a 1.0 (comportement binaire).
        try:
            from . import serveur_lora as _srv
            if _srv._actifs():
                scores = analyse.get("scores") or {}
                poids = {cat: s for cat, s in scores.items()
                         if cat in _srv._IDS and s >= 0.15}
                if not poids:
                    poids = {categorie: 1.0}
                elif categorie not in poids:
                    poids[categorie] = max(max(poids.values()), 0.5)
                _srv.activer_mixte(poids)
        except Exception as e:
            LOG.info("[serveur_lora] mix route (%s) -> poids defaut", e)
        try:
            from .llama_cerveau import llm_charge as _llm_charge
            _llm = _llm_charge()
            if _llm is not None:
                adaptateurs.appliquer(_llm, categorie)
        except Exception as e:
            LOG.info("[adaptateurs] hot-swap impossible (%s) -> cerveau brut", e)
        reponse = self._generer(question_envoyee, contexte_web, formule,
                                riche=riche, contexte_faits=contexte_faits,
                                personnalite=personnalite,
                                categorie=categorie,
                                complexe=riche)
        # AGENT 3 (redacteur) : les tics de langage du 1B sont retires
        try:
            reponse = agents.nettoyer_style(reponse)
        except Exception as e:
            LOG.info("[agents] nettoyage impossible (%s) -> texte brut", e)
        # VERIFICATION OUTILLEE (pattern CRITIC) : preuve web avant livraison
        verifiee = False
        if self._a_besoin_verification(question, contexte_web, reponse):
            reponse = self._verifier_au_web(question, reponse)
            verifiee = True
            # RECONSOLIDATION : la reponse verifiee remplace l'ancienne au
            # cache (sinon une reponse fausse d'avant reste collée a vie)
            filtre_instantane.mettre_a_jour(question, reponse)
        # APPRENTISSAGE DU GRAPHE : les reponses verifiees web nourrissent
        # le graphe de faits (extraction conservatrice) — le systeme devient
        # plus savant a chaque question factuelle confirmee.
        if verifiee:
            try:
                graphe_faits.apprendre_de_reponse(reponse, verifiee_web=True)
            except Exception as e:
                LOG.info("[graphe] apprentissage impossible : %s", e)
        # memorise pour les futures questions (cache semantique + conversation)
        filtre_instantane.enregistrer(question, reponse)
        self._historique.append({"role": "user", "content": question})
        self._historique.append({"role": "assistant", "content": reponse})
        return {"question": question, "experts": experts, "analyse": analyse,
                "cerveau_choisi": "llama-3.2-1b", "contexte_web": contexte_web,
                "formule": formule, "erreur_pgs": self.derniere_erreur,
                "reponse": reponse}
