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
import time

# Questions d'identite (« qui es tu », « qui t'a cree ») : reponse non
# hallucinable — tiree d'une constante, jamais des poids du 1B.
_IDENTITE = "Aura, l'assistant local 100% offline de Simon. J'assemble un " \
    "cerveau 1B, des maths exactes, des faits verifies web et 7 agents " \
    "cognitifs. Mon createur est Simon (Simonc44)."
_MOTS_IDENTITE = ("qui es tu", "qui es-tu", "qui t'a", "qui t'a cree",
                  "qui t'a cree", "ton createur", "ton nom", "tu t'appelles",
                  "tu es qui", "que sais tu faire", "que peux tu faire",
                  "tu sais faire quoi")

# QUESTIONS META SUR L'IA (« es-tu une ia », « aussi puissant qu'un 8B ») :
# ni une tache de redaction ni un fait a verifier. Sans ce garde-fou,
# _est_complexe (>= 9 mots) declenchait le mode dissertation (plan +
# 3 sections + double passe = ~1400 tokens de CPU, soit ~150 s) sur une
# question fermee — d'ou la sortie hors-sujet repetee observee en prod.
# Signature volontairement PRECISE : « es tu », « qui es tu », paroles
# diregees AU modele + tailles de modeles (8B, GPT...). Un essai sur la
# « puissance de la technologie » ne tombe PAS dedans (peut rester riche).
_RE_META_IA = re.compile(
    r"(\bes[-\s]?tu\b|\bt'es[-\s]?tu\b|\bqui es[-\s]?tu\b|"
    r"\best[-\s]?ce que tu (es|sois|puisses|saches|comprennes)\b|"
    r"\btu es\s+(?:une?\s+)?(ia|intelligence artificielle|robot|modele|"
    r"assistant|conscient|intelligent|capable)\b|"
    r"\bton modele\b|\bton algorithme\b|\bton architecture\b|"
    r"\bta puissance\b|\btes capacites\b|\btes performances\b|"
    r"\bce que tu (peux|sais|connais|comprends)\b|"
    r"\ben tant qu'ia\b|\ben tant que ia\b|"
    r"\bcomment tu (fonctionnes|marches|es construit|travailles)\b|"
    r"\bintelligence artificielle\b|"
    r"\b\d{1,3}[\s-]?b\b|\bgpt[-\s]?\d|\bgpt\b|\bchatgpt\b|\bopenai\b|"
    r"\bllama[-\s]?\d|\bmistral\b|\bdeepseek\b|\bclaude[-\s]?\d|\bgemini\b)",
    re.IGNORECASE)

# Reponse-constante pour les questions de capacite (0 s, zero hallucina-
# tion possible, meme esprit que _IDENTITE) : honnete et technique.
_CAPACITE = (
    "Non : je ne suis pas un modele 8B. Je suis Llama 3.2 1B — un "
    "milliard de parametres — et je tourne EN LOCAL sur le CPU de ce "
    "PC, sans internet ni carte graphique. Un 8B a environ 8 fois plus "
    "de parametres : il est plus fin en nuance, en raisonnement long et "
    "en langues rares, mais il exige une GPU. En contrepartie je suis "
    "immediat sur les questions courtes, 100% hors-ligne (aucune donnee "
    "ne sort de ta machine) et je m'appuie sur des experts symboliques "
    "(maths exactes, faits verifies) qui eux ne se trompent pas. Dis-moi "
    "ton besoin et je te dirai si je suffis — ou s'il te faut plus gros.")


# Question qui attend une VALEUR/UN FAIT court (« quel age a Alice »,
# « quel est le prix du carburant ») : la dissertation multi-pass (75-86 s
# mesures au banc) y est du gachis — on reste en reponse courte.
_RE_REPONSE_COURTE = re.compile(
    r"(\bquel\s+(age|est|sont|prix|nombre|est-ce)|\bquelle?\s+(est|sont|la "
    r"capitale|la reponse)|\bcombien\b|\bqui (est|a|trouve)|"
    r"\bo[ùu] se trouve\b|\bquand (est-ce|a-t-il|est ne)\b|\bprix de\b)",
    re.IGNORECASE)

# Refus INTEGRAL (le modele declina puis se tait) : ne doit jamais etre
# mis en cache ni livre comme si c'etait une reponse. S'il y a une vraie
# suite derriere, c'est un simple tic de tete -> retire.
_RE_REFUS = re.compile(
    r"^\s*(je suis desole|je suis desolee|je ne peux pas|"
    r"je ne suis pas en mesure|desole, mais|pardon, mais)", re.IGNORECASE)


def _est_refus(texte: str) -> bool:
    """Refus court = pas une reponse (refus long avec suite = garde-fou
    legitime, cf. filet hors-ligne)."""
    return bool(texte and _RE_REFUS.search(texte) and len(texte) < 400)


def _retirer_refus_tete(texte: str) -> str:
    """Si la reponse REFUSE puis repond quand meme, coupe le refus de tete
    (« Je suis desole, mais... » suivi d'une vraie analyse = tic parasite)."""
    if not texte or not _RE_REFUS.search(texte):
        return texte
    phrases = re.split(r"(?<=[.!?])\s+", texte.strip())
    if len(phrases) < 2:
        return texte
    reste = " ".join(phrases[1:]).strip()
    # on ne coupe que si la suite est substantielle (sinon on garde le
    # garde-fou : un refus honnete reste un refus honnete)
    return reste if len(reste) >= 200 else texte


def _est_meta_ia(question: str) -> bool:
    """Question adressee AU modele sur lui-meme (capacites, taille, identite)."""
    return bool(_RE_META_IA.search(question or ""))

# Definition/savoir (« qu'est-ce que X », « explique X ») : reponses que le
# 1B invente avec aplomb — le juge d'un petit modele est mauvais, donc on
# ANCRE : contexte web obligatoire (ou graphe de faits) avant de repondre.
_RE_DEF = re.compile(
    r"\b(qu'?est.?ce.?que|qu'?est.?ce.?qu'|c'?est.?quoi|qu'elle? est la "
    r"definition|definis|explique(.?moi)?|montre.?moi)\b", re.IGNORECASE)
_RE_PHRASE_MIN = re.compile(r"\b(theoreme|theorie|loi|principe|concept|"
    r"formule|definition|signifie|histoire de| fonctionne)", re.IGNORECASE)
from concurrent.futures import ThreadPoolExecutor

from . import expert_symbolique, memoire_web, autoamelioration, filtre_instantane, rag_binaire
from . import llama_cerveau, raisonneur, graphe_faits
from . import logique as solveur_logique
from . import potcode
from . import arbre_reflexion
from . import agents
from . import adaptateurs
from . import calcul_verbal
from . import tri_transitif
from . import fast_lang
from . import typographie as _typo
from .memoire_conversation import MemoireConversation
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

# BUDGET TEMPS GLOBAL : les etapes FACULTATIVES (double passe de style,
# verification web) ne demarrent plus si le pipeline a deja depasse ce
# plafond — un passage long ne doit pas se voir empiler 2 passes de +.
# Ce garde-fou ne coupe pas une generation en cours (impossible sans tuer
# le process llama), il empeche surtout l'empilement qui a produit les
# 153 s observes. AURA_BUDGET_S=0 pour toujours les executer.
_BUDGET_S = float(os.environ.get("AURA_BUDGET_S", "45"))

# MAJORITE D'EXPERTS (self-consistency) : sur les terrains EXACTS (puzzles
# d'ages, enigmes de logique), le meme cerveau est echantillonne N fois et
# CHAQUE essai passe un solveur deterministe — la reponse retenue est celle
# qui recueille le plus de voix VERIFIEES (un essai casse ne vote pas).
# AURA_MAJORITE_N=1 coupe (ancien comportement : un seul essai).
# Borne a 3 : au-dela le gain mesurable est negligeable face au cout CPU.
_MAJORITE_MAX = 3


def _nb_majorite(t0: float | None = None) -> int:
    """Nombre d'essais a lancer (1 si coupe ou budget epuise)."""
    try:
        n = int(os.environ.get("AURA_MAJORITE_N", "3"))
    except ValueError:
        n = 3
    n = max(1, min(n, _MAJORITE_MAX))
    if n > 1 and t0 is not None and _BUDGET_S > 0 \
            and time.time() - t0 >= _BUDGET_S:
        return 1                     # budget epuise : plus d'essai supp.
    return n


def _cle_reponse_pot(brut: str, dernier: float | None) -> str:
    """Cle de vote d'un essai PoT : le nombre de la REPONSE, sinon texte.

    Deux essais qui annoncent « 11 ans » et « REPONSE : 11 » doivent voter
    POUR LA MEME chose — la cle est le chiffre extrait de la reponse (a
    defaut, le dernier resultat AST, puis le texte normalise).
    """
    m = re.search(r"REPONSE\s*:\s*(.+)", brut, re.IGNORECASE)
    texte = m.group(1) if m else (raisonneur._fmt(dernier)
                                  if dernier is not None else "")
    texte = " ".join(texte.lower().split())
    chiffres = re.search(r"\d+(?:[.,]\d+)?", texte)
    return chiffres.group(0).replace(",", ".") if chiffres else texte


def _contient_valeur(texte: str | None, cibles: list[str]) -> bool:
    """Vrai si le texte exprime deja l'une des valeurs attendues.

    Garde-fou de la synthese PoT : « 9 » compte vu dans « Alice a 9 ans »
    comme dans « ... = 9 » (meme tolérance que le banc de qualité).
    """
    if not texte:
        return False
    return any(c and c in texte for c in cibles)


# Relance de la synthese PoT : le 1B re-derive parfois en ignorant le bloc
# verifie (mesure : « 5 » / « 7 » repondus alors que la derniere etape AST
# valait 9). Un systeme STERILE (« tu ne calcules jamais ») portait la
# reprise a 3/3 au banc, contre 0/2 sans systeme.
_SYSTEME_RECOPIE = (
    "Tu es un assistant qui RECOPIE des valeurs deja calculees par un "
    "programme. Tu ne recalcules et ne deduis JAMAIS. Reponds en UNE "
    "seule phrase qui contient la valeur exacte qui te est donnee.")


class Aura1B:
    """Systeme MoE autonome : cerveau Llama + experts PGS / web."""

    def __init__(self):
        self.expert = expert_symbolique.ExpertSymbolique()
        self.derniere_formule = ""
        self.derniere_erreur = None
        self._routeur = RouteurIntelligent()
        self._llama = None
        # MEMOIRE DE CONVERSATION (RLM MIT + fenetre glissante + etat
        # evenementiel) : l'historique complet vit dans un FICHIER externe
        # consultable par outil (recherche arriere), seuls les 4 derniers
        # tours bruts alimentent la fenetre, et l'etat civil (nom,
        # possessions...) est extrait par regex en JSON compact. Fin de
        # l'historique brut injecte en masse dans le 1B.
        self._historique = MemoireConversation()
        self._chaine_faits: list = []        # chaine graphe de la question
        self._derniere_categorie = "general"
        # Dernier arbre de reflexion construit (PoT) : trace exploitable
        # pour le debug/les tests (meme esprit que derniere_formule).
        self._dernier_arbre: arbre_reflexion.ArbreReflexion | None = None

    # -- memoire de conversation -------------------------------------------

    def reinitialiser_conversation(self):
        """Oublie la conversation en cours (nouveau sujet).

        Fenetre + etat civil remis a zero ; le fichier RLM est PRESERVE :
        la memoire longue reste accessible via l'outil de recherche.
        """
        self._historique.vider()

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

    @staticmethod
    def _extraire_chaine(contexte_faits: str) -> list[tuple[str, str]]:
        """Extrait [(sujet, objet)] du contexte graphe (lignes '- s : r o')."""
        chaine = []
        for ligne in (contexte_faits or "").splitlines():
            ligne = ligne.strip().lstrip("- ")
            if " : " not in ligne:
                continue
            gauche, _, droite = ligne.partition(" : ")
            gauche = gauche.strip()
            droite = droite.split(" ", 1)[-1].strip()   # saute la relation
            if gauche and droite:
                chaine.append((gauche, droite))
        return chaine

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
                 categorie: str = "", complexe: bool = False,
                 est_def: bool = False,
                 max_tokens: int | None = None,
                 think: bool | None = None) -> str:
        if self._llama is None:
            from .llama_cerveau import generer as llama_generer
            self._llama = llama_generer
        # MIMETISME (few-shot) : une brique d'exemples « calibre grand
        # modele » guide le style — dans le PROMPT SYSTEME, jamais collee
        # a la question : le 1B recopiait les exemples (« Exemples de
        # style attendus... ») au lieu de repondre a la question.
        # Mode riche : pas de brique (generer_riche a deja son guidage
        # de style propre, doublon inutile).
        systeme_final = personnalite
        try:
            brique = agents.exemple_style(question, categorie)
            if brique and not riche:
                systeme_final = (f"{personnalite}\n\n{brique}"
                                 if personnalite else brique)
        except Exception:
            pass
        # DETECTEUR DE CONCEPTS : la question releve-t-elle d'un concept
        # abstrait connu (bibliotheque MiniLM) ? Alors un bloc LEXIQUE est
        # ajoute au prompt systeme : le 1B recopie une formulation exacte
        # et calibree au lieu d'improviser sur un mot qu'il ne maitrise
        # pas. Desactive par AURA_CONCEPTS=0.
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
                complexe=complexe,
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
                complexe=complexe,
                max_tokens=max_tokens)
        # AURA_SERVEUR=1 : le Dynamic Compute (reflection masquee) remplace
        # le multi-pass in-process (qui chargerait le cerveau a double).
        if riche and serveur_lora._actifs():
            return self._llama(question, contexte_web, formule,
                               historique=self._historique,
                               contexte_faits=contexte_faits,
                               systeme=self._historique.bloc_systeme(personnalite),
                               categorie=categorie,
                               complexe=True,
                               max_tokens=max_tokens,
                               # think pose SEULEMENT actif (tests/mocks
                               # a signature etroite : kwarg absent si off)
                               **({"think": think} if think else {}))
        if riche:
            from .llama_cerveau import generer_riche
            return generer_riche(question, contexte_web, formule,
                                 historique=self._historique)
        # FILET FINAL (anti-hallucination) : question de definition/savoir
        # SANS aucun ancrage (web injoignable, graphe muet, pas d'expert
        # special) -> on n'invente pas : garde-fou honnete + contexte du
        # graphe s'il existe. Un « je ne peux pas verifier » vaut mieux
        # qu'une definition fausse dite avec assurance. est_def est
        # calcule sur la question ORIGINALE (jamais sur la question +
        # brique few-shot, cf. executer_detaille).
        if (est_def and not contexte_web and not contexte_faits
                and not self._special_traite):
            return ("Je ne peux pas verifier cette definition en ce moment "
                    "(hors-ligne et absente de ma base de faits). Pour ne "
                    "pas t'inventer une reponse fausse, je prefere m'abstenir. "
                    "Reessaie avec internet, ou apprends-moi ce fait : "
                    "graphe_faits.ajouter(...).")
        return self._llama(question, contexte_web, formule,
                           historique=self._historique,
                           contexte_faits=contexte_faits,
                           systeme=self._historique.bloc_systeme(systeme_final),
                           categorie=categorie,
                           complexe=complexe,
                           max_tokens=max_tokens,
                           **({"think": think} if think else {}))

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
        Les questions ABSTRAITES (« peut on etre heureux sans etre libre »)
        sont souvent courtes : elles sont detectees par la forme meme de
        la question (peut on, devrait on, y a t il...) — c'est le terrain
        de la dissertation TAS (these/antithese/synthese).
        """
        q = question.lower()
        # META-IA : une question adressee AU modele n'est jamais une
        # dissertation, meme longue (cf. _RE_META_IA) — sinon le mode
        # riche part en plan + 3 sections pour une question fermee.
        if _est_meta_ia(question):
            return False
        # REPONSE COURTE attendue (age, prix, capitale...) : le multi-pass
        # (75-86 s mesures au banc) y est du gachis.
        if _RE_REPONSE_COURTE.search(question):
            return False
        if len(q.split()) >= 9:                      # question developpee
            return True
        return any(m in q for m in (
            "explique", "analyse", "compare", "pourquoi", "discute",
            "redige", "essai", "opinion", "avis", "argumente", "dissertation",
            # formes abstraites courtes : « peut on X (sans/avec/sans que) »
            "peut on", "peut-on", "pourrait on", "devrait on", "faut il",
            "faut-il", "y a t il", "est il possible", "est-il possible",
            "existe t il", "a t on le droit"))

    # -- raisonnement Program-of-Thoughts -----------------------------------

    @staticmethod
    def _est_puzzle(question: str) -> bool:
        """Puzzle d'ages/relations : le terrain ou PoT rapporte le plus."""
        try:
            return raisonneur.est_puzzle(question)
        except Exception:
            return False

    @staticmethod
    def _auto_think(question: str) -> bool:
        """Puzzle d'ages ou enigme de logique -> brouillon <thinking> utile.

        Le 1B derape justement sur ces terrains : une passe de reflexion
        masquee (separee par separer_reflexion, jamais montree a
        l'utilisateur) coute ~2x de tokens mais est le levier de
        profondeur restant sur un cerveau 1B sans reflexion native.
        """
        return (Aura1B._est_puzzle(question)
                or bool(_ENIGME_LOGIQUE.search(question.lower())))

    # -- enigme logique (mini-SAT pur Python) -------------------------------

    def _resoudre_par_logique(self, question: str,
                              t0: float | None = None) -> str | None:
        """Le 1B formalise (entites/domaine/dits), le solveur deduit exactement.

        MAJORITE D'EXPERTS : N formalisations independantes sont echantillon-
        nees et CHAQUE vote avec sa SOLUTION — une formalisation fausse est
        sortie du lot par les autres ; ex aequo -> la premiere solution
        unique vue (ancien comportement, jamais plus strict qu'avant).
        UNE relance de correction (self-debug) si le 1er essai est rejete.
        Aucun essai exploitable -> None (chemin normal).
        """
        n = _nb_majorite(t0)
        voix: dict[frozenset, int] = {}
        details: dict[frozenset, tuple] = {}
        ordre: list[frozenset] = []
        relance = False
        try:
            for i in range(n):
                # cadrage « Enigme : … / Formalisme : » : mesuré en réel, c'est
                # celui qui fait respecter le format au 1B (noms exacts)
                brut = llama_cerveau.generer(
                    f"Enigme :\n{question}\n\nFormalisme :",
                    systeme=solveur_logique._SYSTEME_LOGIQUE,
                    max_tokens=300)
                try:
                    puzzle = solveur_logique.parser_puzzle(brut)
                except ValueError as e:
                    if relance or i > 0:
                        # la relance self-debug est unique (cout borne) :
                        # les essais suivants sont simplement eclates
                        LOG.info("[logique] formalisation %d/%d rejetee (%s)",
                                 i + 1, n, e)
                        continue
                    relance = True
                    LOG.info("[logique] formalisation rejetee (%s) -> 1 relance", e)
                    second = llama_cerveau.generer(
                        question + "\n\nTa reponse precedente etait :\n" + brut
                        + "\n\nElle a ete REJETEE car : " + str(e)
                        + "\nReponds a nouveau, STRICTEMENT selon le format, "
                          "avec les vrais noms de l'enigme et SANS inventer de "
                          "contrainte.",
                        systeme=solveur_logique._SYSTEME_LOGIQUE,
                        max_tokens=300)
                    try:
                        puzzle = solveur_logique.parser_puzzle(second)
                    except ValueError as e2:
                        LOG.info("[logique] relance rejetee (%s)", e2)
                        continue
                r = solveur_logique.resoudre(puzzle)
                if r["statut"] != "unique":
                    LOG.info("[logique] statut %s (essai %d/%d)",
                             r["statut"], i + 1, n)
                    continue
                cle = frozenset(tuple(sorted(s.items()))
                                for s in r["solutions"])
                if cle not in voix:
                    voix[cle] = 0
                    details[cle] = (puzzle, r["solutions"])
                    ordre.append(cle)
                voix[cle] += 1
            if not voix:
                LOG.info("[logique] aucun essai unique -> chemin normal")
                return None
            # majorite si elle existe, sinon la 1re solution unique vue
            cle_gagnante = max(ordre, key=lambda c: voix[c])
            if len(voix) > 1:
                LOG.info("[logique] vote : %d solutions divergentes, "
                         "gagnante %d/%d voix", len(voix),
                         voix[cle_gagnante], n)
            puzzle_gagnant, solutions_gagnantes = details[cle_gagnante]
            verification = solveur_logique.bloc_verification(
                puzzle_gagnant, solutions_gagnantes)
            finale = llama_cerveau.generer(question, contexte_web=verification,
                                           max_tokens=150)
            LOG.info("[logique] enigme resolue exactement (%d essai(s))", n)
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
        if _ENIGME_LOGIQUE.search(question.lower()):
            rep = self._resoudre_par_logique(question, t0=t0)
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

    def _resoudre_par_pot(self, question: str,
                          t0: float | None = None) -> str | None:
        """Passe PoT : le 1B ecrit les etapes, l'AST verifie chaque calcul.

        MAJORITE D'EXPERTS + ARBRE DE REFLEXION : chaque essai echantillonne
        devient une BRANCHE de l'arbre (structures MCTS), notee par l'AST
        etape par etape (validee 1.0 / cassee 0.0) ; les branches valides
        votent sur la reponse finale — la majorite gagne, ex aequo -> la
        meilleure branche. La branche gagnante est reinjectee a la synthese
        comme brouillon deja prouve (branche_vers_prompt).

        Renvoie None si AUCUN essai ne produit >= 2 etapes calculables —
        dans ce cas on retombe sur le chemin normal (jamais pire qu'avant).
        """
        n = _nb_majorite(t0)
        arbre = arbre_reflexion.ArbreReflexion(question=question)
        self._dernier_arbre = arbre
        essais: list = []      # (etapes, resultats, cle de vote, branche)
        for i in range(n):
            try:
                brut = llama_cerveau.generer(question,
                                             systeme=raisonneur._SYSTEME_POT,
                                             max_tokens=280)
            except Exception as e:
                LOG.info("[pot] generation %d/%d impossible : %s", i + 1, n, e)
                break
            etapes = raisonneur.extraire_etapes(brut)
            if len(etapes) < 2:
                LOG.info("[pot] pas assez d'etapes calculables (%d, essai %d/%d)",
                         len(etapes), i + 1, n)
                continue
            branche = arbre.nouvelle_branche()
            resultats: list = []
            valide = True
            for phrase, calcul in etapes:
                try:
                    res = raisonneur.verifier_calculs([(phrase, calcul)])[0]
                    ok = True
                except ValueError:
                    res, ok, valide = None, False, False
                branche.ajouter(f"{phrase} = {calcul}", expert="pot",
                                score=1.0 if ok else 0.0, valide=ok)
                resultats.append(res)
            if not valide:
                continue       # branche KO conservee dans l'arbre (trace)
            essais.append((etapes, resultats,
                           _cle_reponse_pot(brut, resultats[-1]), branche))
        if not essais:
            LOG.info("[pot] aucun essai valide -> chemin normal")
            return None
        # VOTE : la cle = la reponse annoncee (chiffre) ; majorite si elle
        # existe, ex aequo -> l'arbre tranche (branche seriee puis notee).
        voix: dict[str, int] = {}
        for essai in essais:
            voix[essai[2]] = voix.get(essai[2], 0) + 1
        cle_gagnante = max(voix, key=lambda c: voix[c])   # ex aequo -> 1re vue
        candidats = [e for e in essais if e[2] == cle_gagnante]
        # La branche dont le CALCUL AST confirme la reponse votee passe
        # devant (mesure : une etape mal ordonnee faisait dire « 8 » a la
        # synthese alors que le vote portait « 9 »).
        confirmes = [e for e in candidats
                     if e[1] and raisonneur._fmt(e[1][-1]) == cle_gagnante]
        etapes, resultats, cle, branche = max(
            confirmes or candidats,
            key=lambda e: (e[3].serie,
                           e[3].score if e[3].score is not None else -1.0))
        if len(essais) > 1:
            LOG.info("[pot] majorite %d/%d voix vers %r", voix[cle],
                     len(essais), cle)
        LOG.info("[pot] arbre de reflexion :\n%s", arbre.trace())
        verification = raisonneur.construire_verification(etapes, resultats)
        brouillon = arbre_reflexion.branche_vers_prompt(branche)
        if brouillon:
            verification += ("\n\nREFLEXION PROUVEE A METTRE EN FRANCAIS :\n"
                             + brouillon)
        try:
            finale = llama_cerveau.generer(question,
                                           contexte_web=verification,
                                           max_tokens=160)
        except Exception as e:
            LOG.info("[pot] synthese impossible : %s", e)
            return None
        # GARANTIE DE LA VALEUR VERIFIEE : la synthese LLM re-derive parfois
        # en ignorant le bloc verifie (mesure : « 6 » / « 7 » repondus alors
        # que la derniere etape AST valait 9). Une relance explicite, puis —
        # si le modele s'obstine — une phrase construite PAR LE PROGRAMME :
        # la valeur prouvee ne disparait jamais de la reponse.
        cibles = [raisonneur._fmt(resultats[-1])]     # dernier resultat AST
        if cle and cle not in cibles:
            cibles.append(cle)                        # majorite des essais
        if not _contient_valeur(finale, cibles):
            try:
                relance = llama_cerveau.generer(
                    question,
                    systeme=_SYSTEME_RECOPIE,
                    contexte_web=(verification
                                  + f"\nLa valeur exacte est {cibles[0]}. "
                                  "Reponds en UNE seule phrase qui contient "
                                  "cette valeur."),
                    max_tokens=80)
                if _contient_valeur(relance, cibles):
                    finale = relance
            except Exception as e:
                LOG.info("[pot] relance impossible : %s", e)
        if not _contient_valeur(finale, cibles):
            # deux echecs : on livre la reponse PAR LE CODE, en une phrase
            # seule plutot qu'en accumulant deux ages contradictoires
            LOG.info("[pot] synthese sans la valeur verifiee -> reponse du code")
            if (cle and cle != cibles[0]
                    and re.fullmatch(r"\d+(?:[.,]\d+)?", cle)):
                # vote et calcul AST divergent : on annonce la reponse VOTEE
                # (contrat du vote), jamais un chiffre que le vote rejette
                finale = f"Reponse retenue par la majorite des essais : {cle}."
            else:
                finale = raisonneur.phrase_verifiee(etapes, resultats)
        LOG.info("[pot] puzzle resolu : %d etapes verifiees (%d essai(s))",
                 len(etapes), len(essais))
        return finale

    # -- pipeline complet ---------------------------------------------------

    def executer(self, question: str, X=None, y=None) -> str:
        # COUTEAU TYPOGRAPHIQUE (post-processeur regex, < 0,1 ms) : la
        # réponse finale sort propre (« l'arbre », NBSP devant ?!;:,
        # apostrophes typographiques) — l'intérieur des blocs ```code```
        # n'est JAMAIS touché (le code vérifié doit rester exécutable).
        return _typo.hors_code(
            self.executer_detaille(question, X, y)["reponse"])

    def executer_detaille(self, question: str, X=None, y=None) -> dict:
        t0 = time.time()                 # horloge du budget global (cf. _BUDGET_S)
        # ── CYCLE UNIQUE (idea JEV #3) ─────────────────────────────────
        # maths directes + cache semantique + routage en UN passage : la
        # majorite des questions quotidiennes n'ont PAS besoin du LLM.
        instant, analyse = self._decider(question)
        if instant is not None and not (X and y):
            # NETTOYAGE DU CACHE : une reponse sale peut y etre restee d'un
            # run precedent (phrase en double, refus) — vu au banc
            # (logique-ages : refus double servi en 0,1 s).
            instant = llama_cerveau._sans_doublons(instant)
            # une vieille entree de cache peut contenir un libelle de plan
            # (SECTION A REDIGER, PARTIE n :) : il est retire au service.
            instant = llama_cerveau._sans_libelles(instant) or instant
            if _est_refus(instant):
                LOG.info("[niveau0] refus servi par le cache -> pipeline complet")
                instant, analyse = None, self.analyser(question)
            else:
                return {"question": question, "experts": {"instantane"},
                        "analyse": {"methodes": ["niveau0"]},
                        "cerveau_choisi": "niveau0-instantane",
                        "contexte_web": "", "formule": "",
                        "erreur_pgs": None, "reponse": instant}

        # IDENTITE : reponse constante, hors des poids — l'apparence d'Aura
        # ne se negocie pas avec un 1B (question « qui es tu » -> reponse
        # stable et veridique, 0 ms, zero hallucination possible).
        basse = question.lower()
        if any(m in basse for m in _MOTS_IDENTITE):
            return {"question": question, "experts": {"identite"},
                    "analyse": {"methodes": ["identite"]},
                    "cerveau_choisi": "constante",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": _IDENTITE}

        # CAPACITE (meta-IA courte, « es tu aussi puissant qu'un 8B ») :
        # meme logique que l'identite — question fermee adressee au modele,
        # reponse constante verifiee, 0 s de CPU, zero risque de disser-
        # tation hors-sujet ou de boucle de repetition.
        if _est_meta_ia(question) and len(question.split()) <= 14:
            return {"question": question, "experts": {"identite"},
                    "analyse": {"methodes": ["capacite"]},
                    "cerveau_choisi": "constante-capacite",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": _CAPACITE}
        # meta OUVERTE (longue) : elle repond, mais JAMAIS en mode
        # redaction par sections — _est_complexe refuse, et le budget de
        # tokens est borne (reponse courte, pas 1400 tokens de CPU).
        meta_ia = _est_meta_ia(question)

        # ── NIVEAUX 1-2 : experts + LLM ───────────────────────────────
        # VALIDATION AVANT EXECUTION (idea JEV #2 + confiance RouteLLM) :
        # chaque expert est verifie avant d'etre lance.
        experts = self._valider_experts(analyse, X, y)
        # ANCRAGE DES DEFINITIONS : « qu'est-ce que X » / « explique X » ->
        # le web expert est FORCE (si internet est la). Sans contexte, le
        # 1B invente des definitions fausses avec aplomb (« Pythagore » =
        # « plus petit carre commun ») : mieux vaut ancrer que deviner.
        # est_def est calcule sur la question ORIGINALE et reutilise partout
        # (le filet final ne doit JAMAIS tester la question enrichie : la
        # brique few-shot contient « formule », « fonctionne »... qui
        # declencheraient le filet a tort sur des questions innocentes).
        est_def = bool(_RE_DEF.search(question)
                       or _RE_PHRASE_MIN.search(question))
        if est_def:
            experts.discard("general")
            experts.add("web")
        riche = self._est_complexe(question)
        # DETECTION UNIQUE (executes une seule fois) :
        # - est_puzzle pilote le gate RAG et la passe PoT ;
        # - auto_think pilote le brouillon <thinking> a la generation
        #   finale (les puzzles/enigmes qui ont RATE leurs experts
        #   deterministes repartent avec une passe de reflexion masquee).
        est_puzzle = self._est_puzzle(question)
        auto_think = self._auto_think(question)

        # EXPERTS SPECIAUX (logique exacte, code sandboxe) : signatures
        # conservatrices, repli cascade vers le chemin normal. AVANT le
        # fan-out : une enigme de logique n'a pas besoin de DuckDuckGo.
        # CALCUL VERBAL : les petits problemes racontes (« j'ai 3 pommes,
        # j'en mange 1, combien il m'en reste ») n'ont ni expression
        # mathematique ni donnees PGS — sans ce detecteur, le 1B repond
        # un essai hors-sujet (nutrition...). Traduction en calcul exact
        # par regles ; None -> cascade normale. SANS condition riche :
        # un probleme raconte est souvent long (> 9 mots) mais reste un
        # calcul — la longueur ne doit pas le priver de l'exactitude.
        # FAST-LANG (dictionnaire symbolique SQLite, O(1)) : conjugaison
        # et définition connues de la base répondent SANS LLM, sans
        # hallucination possible (< 0,2 ms). Miss -> cascade normale :
        # les définitions inconnues partent vers l'ancrage web, comme avant.
        reponse_lang = fast_lang.moteur.repondre(question)
        if reponse_lang is not None:
            self._special_traite = True
            return self._resultat_special(question, reponse_lang,
                                          "fast-lang")
        verbal = calcul_verbal.resoudre(question)
        if verbal is not None:
            self._special_traite = True
            return self._resultat_special(question, verbal,
                                          "calcul-verbal")
        special = self._expert_special(question, X, y, t0=t0)
        if special is not None:
            self._special_traite = True     # un expert exact a repondu
            return special
        self._special_traite = False

        # RAG BINAIRE (source 2 de la cascade) : le savoir verifie local
        # repond en ~5 ms SANS reseau. Si l'index connait la question
        # (similarite suffisante), on renvoie la reponse extraite telle
        # quelle — zero hallucination possible (rien n'est genere).
        # APRES les experts speciaux : un lookup flou ne doit JAMAIS
        # primer sur une resolution exacte (tri-transitif, SAT) — sinon
        # une enigme « qui est le plus age ? » est servie par l'index au
        # lieu du solveur (regression vue au banc). Les PUZZLES sont
        # egalement exclus : ils se deduisent (PoT + AST), ils ne se
        # documentent pas — l'index n'a rien a y faire avant le raisonneur.
        if not riche and not est_puzzle:
            try:
                rag_rep = rag_binaire.chercher(question)
            except Exception as e:
                LOG.info("[rag] indisponible (%s)", e)
                rag_rep = None
            if rag_rep:
                return {"question": question,
                        "experts": experts | {"rag"},
                        "analyse": analyse, "cerveau_choisi": "rag-binaire",
                        "contexte_web": "", "formule": "",
                        "erreur_pgs": None, "reponse": rag_rep}

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
            # ANCRAGE EXTRACTIF des definitions : le contexte web VERIFIE
            # est la reponse — on le cite tel quel au lieu de le faire
            # reformuler par le 1B (c'est LA que l'hallucination entrait :
            # le modele melangeait faits fournis et memoire interne, cf.
            # « Pythagore = plus petit carre commun »). Extraction =
            # zero generation = zero hallucination possible. SANS condition
            # riche : une definition est un lookup factuel, meme en mode
            # complexe — le style riche n'a pas de sens pour citer des faits.
            if not formule and est_def:
                # dedup : deux extraits de recherche identiques ne doivent
                # pas etre livres cote a cote (phrase repetee a l'ecran)
                reponse_extraite = (
                    "D'apres les sources verifiees en ligne :\n"
                    + llama_cerveau._sans_doublons(contexte_web))
                filtre_instantane.enregistrer(question, reponse_extraite)
                return {"question": question, "experts": experts | {"web"},
                        "analyse": analyse, "cerveau_choisi": "extraction-web",
                        "contexte_web": contexte_web, "formule": formule,
                        "erreur_pgs": None, "reponse": reponse_extraite}
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
        if not (X and y) and not contexte_web and est_puzzle:
            pot = self._resoudre_par_pot(question, t0=t0)
            if pot is not None:
                filtre_instantane.enregistrer(question, pot)
                return {"question": question, "experts": experts | {"pot"},
                        "analyse": analyse, "cerveau_choisi": "pot-1b+ast",
                        "contexte_web": "", "formule": "",
                        "erreur_pgs": None, "reponse": pot}

        # GRAPHE DE FAITS (MiniRAG-lite) : retrouver au lieu de deviner —
        # uniquement des faits VERIFIES web (jamais d'hallucination dedans).
        # La chaine utilisee est memorisee : si la reponse finale est
        # confirmee par un expert symbolique, elle sera RENFORCEE (MLT).
        contexte_faits = "" if contexte_web else \
            graphe_faits.chercher(question)
        self._chaine_faits = self._extraire_chaine(contexte_faits)
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
        # meta-ia ouverte : budget borne (reponse courte, jamais1400 tokens)
        reponse = self._generer(question_envoyee, contexte_web, formule,
                                riche=riche, contexte_faits=contexte_faits,
                                personnalite=personnalite,
                                categorie=categorie,
                                complexe=riche, est_def=est_def,
                                max_tokens=(110 if meta_ia else None),
                                # AUTO-THINK : puzzles/enigmes seulement,
                                # et seulement s'il reste du budget
                                think=(auto_think and (_BUDGET_S <= 0
                                       or time.time() - t0 < _BUDGET_S)))
        # AGENT 3 (redacteur) : les tics de langage du 1B sont retires
        try:
            reponse = agents.nettoyer_style(reponse)
        except Exception as e:
            LOG.info("[agents] nettoyage impossible (%s) -> texte brut", e)
        # GARDE-FOU FINAL (disjoncteur) : JAMAIS une reponse en boucle ne
        # part vers l'utilisateur — on deduplique les phrases repetees ;
        # si c'est une boucle serree (n-grammes), on coupe avant repetition.
        # PUIS tete de refus parasite : « Je suis desole, mais... » suivi
        # d'une vraie suite est coupe (un refus SANS suite reste conserve).
        try:
            # LIBELLES DE PLAN : « SECTION A REDIGER : PARTIE n » recopie
            # par le 1B ne doit JAMAIS arriver a l'utilisateur (banc :
            # riche-puissance KO sur cette balise ecolee).
            reponse = llama_cerveau._sans_libelles(reponse) or reponse
            # PHRASES COPIEES : mot pour mot OU par fragment de 45 signes
            # (« Pourtant, cette question souleve... » puis « En definitive,
            # cette question souleve... » reste une copie malgre son debut).
            sans_dbl = llama_cerveau._sans_doublons(reponse)
            if sans_dbl != reponse:
                LOG.warning("[disjoncteur] phrases copiees retirees en fin "
                            "de chaine")
                reponse = sans_dbl
            if llama_cerveau._en_boucle(reponse):
                reponse = llama_cerveau._couper_boucle(reponse)
                LOG.warning("[disjoncteur] reponse nettoyee en fin de chaine")
            # phrase tronquee en fin de texte (max_tokens) : finition propre
            reponse = llama_cerveau._finir_phrase(reponse)
            reponse = _retirer_refus_tete(reponse)
        except Exception as e:
            LOG.info("[disjoncteur] garde-fou impossible (%s)", e)
        # DOUBLE PASSE (le secret du 1B) : un petit modele est mediocre pour
        # ecrire parfait du premier coup, mais STATISTIQUEMENT EQUIVALENT a un
        # grand modele pour REPERER les erreurs et reformuler un texte existant.
        # Le texte redige est donc soumis a une seconde lecture (critique puis
        # version corrigee) — uniquement si un texte riche a ete produit.
        ecoule = time.time() - t0
        if riche and (_BUDGET_S <= 0 or ecoule < _BUDGET_S):
            try:
                reponse = agents.double_passe(reponse, question)
            except Exception as e:
                LOG.info("[agents] double passe impossible (%s) -> texte brut", e)
        elif riche:
            # deja trop long : 2 passes de style de plus = le chemin qui a
            # fait exploser la latence — on livre le texte deja produit
            LOG.info("[budget] double passe sautee (%.0fs > %.0fs)",
                     ecoule, _BUDGET_S)
        # VERIFICATION OUTILLEE (pattern CRITIC) : preuve web avant livraison
        verifiee = False
        if ((_BUDGET_S <= 0 or time.time() - t0 < _BUDGET_S)
                and self._a_besoin_verification(question, contexte_web,
                                                reponse)):
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
        # MEMOIRE A LONG TERME AUTONOME : une reponse confirmee par un
        # expert symbolique (validation exacte : calcul, logique, code,
        # preuve web) renforce les liens de la chaine de faits utilisee —
        # les schemas de pensee eprouves remontent, les liens inutiles
        # retombent (oubli doux). Aucun re-entrainement requis.
        if self._chaine_faits:
            confirmee = (verifiee or "math" in experts or "pot" in experts
                         or reponse.startswith(("[Logique]", "[Code]")))
            if confirmee:
                for s, o in self._chaine_faits:
                    try:
                        graphe_faits.renforcer(s, o)
                    except Exception:
                        pass
        # NETTOYAGE FINAL AVANT LIVRAISON ET MEMORISATION : quoi qu'aient
        # produit la double passe ou la verification web, aucun libelle de
        # plan, aucune phrase copiee ni fin tronquee ne doit etre servi
        # (ni servi ulterieurement depuis le cache).
        try:
            reponse = llama_cerveau._sans_libelles(reponse) or reponse
            reponse = llama_cerveau._sans_doublons(reponse)
            reponse = llama_cerveau._finir_phrase(reponse)
        except Exception as e:
            LOG.info("[nettoyage final] impossible (%s)", e)
        # memorise pour les futures questions (cache semantique + conversation)
        filtre_instantane.enregistrer(question, reponse)
        self._historique.ajouter("user", question)
        self._historique.ajouter("assistant", reponse)
        return {"question": question, "experts": experts, "analyse": analyse,
                "cerveau_choisi": "llama-3.2-1b", "contexte_web": contexte_web,
                "formule": formule, "erreur_pgs": self.derniere_erreur,
                "reponse": reponse}
