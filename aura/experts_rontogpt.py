"""Experts de RontoGPT : validation avant execution, execution en parallele
et routage des experts speciaux. Mixin herite par RontoGPT (aura/orchestrateur.py)."""
import logging
import os
import threading

from . import memoire_web, filtre_instantane, rag_binaire
from . import chaines_numeriques
from . import problemes_arithmetiques
from . import micro_faits
from . import llama_cerveau
from . import registre_experts
from . import logique as solveur_logique
from . import potcode
from . import apis_publiques
from . import titulaires
from . import calcul_achat
from . import calcul_verbal
from . import calcul_taux
from . import tri_transitif
from . import fast_lang
from .regles_reponse import _est_refus
from .detection_demande import _DEMANDE_CODE_SANDBOX, _ENIGME_LOGIQUE

LOG = logging.getLogger("aura.orchestrateur")


# Seuil de confiance (style RouteLLM) : sous ce score max du classifieur,
# on ne fait pas confiance a son choix -> chemin sur 'general'.
_SEUIL_CONFIANCE = float(os.environ.get("AURA_SEUIL_CONFIANCE", "0.50"))


class ExpertsMixin:
    """Routage et execution des experts (methodes de RontoGPT)."""

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

    def _executer_experts(self, experts, question, X, y):
        """Lance les experts en parallele (web ∥ PGS ∥ API publiques), un
        aller-retour.

        Deux taches -> deux threads (le reseau et le CPU se recouvrent) ;
        une seule -> appel direct, zero surcout. Un expert qui echoue
        ne bloque jamais l'autre : chaque tache est isolee, echec = chaine
        vide. Renvoie (contexte_web, formule) ; le contexte API est mis
        de cote dans self._contexte_api (consomme ensuite par le pipeline).
        """
        self._contexte_api = ""
        taches = {}
        # hors_web : demande de code ou enigme de logique (fix bugs 3 et 4) —
        # rien a chercher sur internet, ni DuckDuckGo ni API publique. Pose par
        # _executer_detaille (attribut, pour ne pas changer la signature).
        hors_web = bool(getattr(self, "_hors_web", False))
        if "web" in experts and not hors_web:
            llama_cerveau.publier_flux("Recherche sur le web…",
                                       llama_cerveau.PHASE_ETAPE)
            requete = self._requete_web(question)
            taches["web"] = (memoire_web.chercher_iteratif, (requete,))
        if "math" in experts:
            taches["pgs"] = (self.resoudre_numerique, (X, y))
        # API PUBLIQUES (liste public-apis) : meteo, devises, definitions,
        # livres, geocodage — sans cle, stdlib. La detection est une regex
        # LOCALE (zero octet si hors sujet) et la tache part en parallele :
        # sa latence se cache sous celle de DuckDuckGo.
        if not hors_web and apis_publiques.detecte(question):
            taches["api"] = (apis_publiques.chercher, (question,))

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
        self._contexte_api = resultats.get("api", "")
        return resultats.get("web", ""), resultats.get("pgs", "")

    # -- cerveau ------------------------------------------------------------

    def _expert_special(self, question: str, X, y,
                        t0: float | None = None) -> dict | None:
        """Routage des experts speciaux, dans l'ordre de specialisation.

        Donnees numeriques -> PGS direct (pas de PoT/logique). Renvoie le
        dict de reponse si un expert a tranché, sinon None.
        """
        if X and y:
            return None
        # TRI TRANSITIF (le plus specialise d'abord) : « A plus age que B,
        # B plus age que C » — resolution deterministe, zero LLM. Si le
        # parseur n'extrait pas une chaine solide (>= 2 relations), None
        # et le mini-SAT prend le relais.
        rep_tri = tri_transitif.repondre(question)
        if rep_tri is not None:
            return self._resultat_special(question, rep_tri, "tri-transitif")
        # CODE D'ABORD (fix bug 3) : une demande de programmation est VERROUILLEE
        # sur la sandbox locale (potcode) — jamais sur l'internet, jamais sur
        # le solveur logique. Contrat de la sandbox : fonction `resoudre`, sans
        # import ni E/S ; une demande hors contrat (JavaScript, fichiers...) est
        # rendue au cerveau SANS web (cf. le verrou de _executer_detaille).
        if potcode.est_demande_code(question):
            if (potcode.demande_sandbox_python(question)
                    or _DEMANDE_CODE_SANDBOX.search(question.lower())):
                rep = self._resoudre_par_code(question, mode_auto=True)
                if rep is not None:
                    return self._resultat_special(question, rep,
                                                  "potcode-1b+sandbox")
            return None
        # ENIGMES (fix bug 4) : detection forte (chevaliers/menteurs, attribution
        # de postes/couleurs/rangs entre >= 3 personnes) -> le 1B n'extrait que
        # le formalisme, le mini-SAT deduit. Si l'extraction echoue, on le DIT
        # au lieu de confier la consigne brute au 1B.
        enigme = solveur_logique.est_enigme(question)
        if enigme or _ENIGME_LOGIQUE.search(question.lower()):
            rep = self._resoudre_par_logique(question, t0=t0)
            if rep is not None:
                return self._resultat_special(question, rep, "logique-1b+sat")
            if enigme:
                return self._enigme_non_formalisee(question)
        if _DEMANDE_CODE_SANDBOX.search(question.lower()):
            rep = self._resoudre_par_code(question, mode_auto=True)
            if rep is not None:
                return self._resultat_special(question, rep, "potcode-1b+sandbox")
        return None

    @staticmethod
    def _enigme_non_formalisee(question: str) -> dict:
        """Enigme reconnue mais non formalisee : refus honnete, NON memorise."""
        message = (
            "J'ai reconnu une énigme de logique, mais je n'ai pas réussi à la "
            "formaliser de façon fiable (personnes, valeurs possibles, "
            "contraintes). Plutôt que de deviner, je préfère ne pas répondre : "
            "reformule-la en donnant les noms, la liste des valeurs possibles "
            "(postes, couleurs, rôles…) puis chaque indice sur sa propre ligne.")
        return {"question": question, "experts": {"logique"},
                "analyse": {"methodes": ["expert_special"]},
                "cerveau_choisi": "logique-non-formalisee",
                "contexte_web": "", "formule": "", "erreur_pgs": None,
                "reponse": message}

    def _resultat_special(self, question: str, reponse: str, cerveau: str) -> dict:
        filtre_instantane.enregistrer(question, reponse)
        return {"question": question, "experts": {cerveau.split("-")[0]},
                "analyse": {"methodes": ["expert_special"]},
                "cerveau_choisi": cerveau, "contexte_web": "", "formule": "",
                "erreur_pgs": None, "reponse": reponse}

    # -- registre d'experts ------------------------------------------------
    # L'ORDRE vient de aura/registre_experts.py (declare + verifie). Ici on
    # ne trouve que le BRANCHEMENT : quel code appeler, et sous quelle
    # garde. Le test tests/test_registre_experts.py exige que CHAQUE expert
    # declare ait les deux, sinon un expert du registre disparaitrait en
    # silence.

    def _appel_expert(self, nom: str, question: str, X, y, t0: float):
        """Appelle l'expert `nom`. None = il ne sait pas repondre."""
        if nom == "fast-lang":
            return fast_lang.moteur.repondre(question)
        if nom == "calcul-achat":
            return calcul_achat.resoudre(question)
        if nom == "calcul-verbal":
            return calcul_verbal.resoudre(question)
        if nom == "calcul-taux":
            return calcul_taux.resoudre(question)
        if nom == "chaines-numeriques":
            return chaines_numeriques.resoudre(question)
        if nom == "problemes-arithmetiques":
            return problemes_arithmetiques.resoudre(question)
        if nom == "micro-faits":
            return micro_faits.resoudre(question)
        if nom == "experts-speciaux":
            return self._expert_special(question, X, y, t0=t0)
        if nom == "titulaires":
            try:
                return titulaires.resoudre(question)
            except Exception as e:      # reseau, JSON : jamais bloquant
                LOG.info("[titulaires] indisponible (%s)", e)
                return None
        if nom == "rag-binaire":
            try:
                return rag_binaire.chercher(question)
            except Exception as e:      # index absent/corrompu : jamais bloquant
                LOG.info("[rag] indisponible (%s)", e)
                return None
        if nom == "cache-semantique":
            return filtre_instantane.repondre_cache(question)
        # Echec BRUYANT : un expert declare au registre sans branchement est
        # un bug, pas un cas a ignorer silencieusement.
        raise KeyError(f"expert declare au registre mais non branche : {nom}")

    @staticmethod
    def _expert_applicable(nom: str, ctx: "registre_experts.Contexte") -> bool:
        """Garde de l'expert `nom` (cf. registre_experts.GARDES).

        Par defaut un expert est applicable : les solveurs exacts n'ont pas
        de condition (c'est le principe). Seuls les etages a lookup ont une
        raison de s'abstenir.
        """
        if nom == "rag-binaire":
            # une demande de code n'est pas un lookup documentaire (fix bug 3)
            return (not ctx.est_puzzle
                    and not memoire_web.a_besoin_web(ctx.question)
                    and not potcode.est_demande_code(ctx.question))
        if nom == "cache-semantique":
            return (not ctx.a_donnees
                    and not memoire_web.a_besoin_web(ctx.question))
        return True

    def _resultat_expert(self, nom: str, rep, question: str, analyse,
                         experts) -> dict | None:
        """Normalise la reponse de `nom`. None = passer a l'expert suivant.

        Deux contrats particuliers, conserves tels quels :
          * `experts-speciaux` renvoie DEJA un dict complet ;
          * `cache-semantique` doit etre nettoye (doublons, libelles de
            plan recopies) et un REFUS servi par la memoire ne doit pas
            sortir — il ne repond pas a la question.
        """
        if nom == "experts-speciaux":
            return rep
        if nom == "cache-semantique":
            memo = llama_cerveau._sans_doublons(str(rep))
            memo = llama_cerveau._sans_libelles(memo) or memo
            # Recitation de prompt memorisee : elle ne vaut pas une reponse.
            # Si tout le cache etait du bavardage, on rend le tour au pipeline
            # au lieu de le servir (une entree « Dis-moi : ... (WEB / BASE DE
            # FAITS) » a ete servie telle quelle, mesure du 12 oct. 2026).
            memo = llama_cerveau._sans_parrotage(memo)
            if not memo.strip() or _est_refus(memo):
                LOG.info("[niveau0] refus servi par le cache -> pipeline")
                return None
            return {"question": question, "experts": experts | {"memoire"},
                    "analyse": analyse, "cerveau_choisi": nom,
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": memo}
        if nom == "titulaires":
            # PAS de filtre_instantane.enregistrer : un titulaire change avec
            # les elections, le cache semantique le servirait perime a des
            # reformulations. L'expert est relu a chaque fois (cache HTTP
            # de 10 min dans apis_publiques._json).
            return {"question": question, "experts": experts | {"web"},
                    "analyse": analyse, "cerveau_choisi": nom,
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": str(rep)}
        if nom == "rag-binaire":
            return {"question": question, "experts": experts | {"rag"},
                    "analyse": analyse, "cerveau_choisi": nom,
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": str(rep)}
        return self._resultat_special(question, str(rep), nom)

    # -- porte de clarification (aura/clarification.py) --------------------
    # Position DECLAREE dans registre_experts.PORTES (« apres-experts »).
    # Cette porte ne repond pas : elle rend la question a l'utilisateur en
    # demandant ce qui manque. Elle est donc placee APRES la derniere couche
    # qui repond (sinon un calcul exact se verrait opposer « precise ta
    # demande ») et AVANT la generation (demander coute moins cher qu'une
    # reponse inventee : le 1B, prive de sujet, ne dit jamais « je ne sais
    # pas de quoi tu parles » — il redige).
