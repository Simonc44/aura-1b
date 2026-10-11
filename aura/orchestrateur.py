"""Orchestrateur autonome : AUCUNE dependance externe (pas Ollama, pas API).

Cerveau local configurable (`aura.toml` / `AURA_GGUF`) :
- Gemma-3-1B-it par defaut dans la configuration courante (llama.cpp),
  instruction-tuned, bon FR + EN,
  flash attention + KV cache q8_0 + memoire de conversation multi-tours

Plus :
- PGS (gplearn) : formules mathematiques exactes, erreur 0
- DuckDuckGo : faits en temps reel
- Auto-amelioration : corrections injectees dans le prompt

Le routeur (TF-IDF + LogReg) choisit les experts a activer ; Llama synthetise.

Cycle de decision inspire de JEV ultrafast (browser-use) :
- niveau 0 exact d'abord ; le routeur ne tourne que si le niveau 0 ne repond pas
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
_IDENTITE = "RontoGPT, l'assistant local 100% offline de Simon. J'assemble un " \
    "cerveau 1B, des maths exactes, des faits verifies web et 7 agents " \
    "cognitifs. Mon createur est Simon (Simonc44)."
_MOTS_IDENTITE = ("qui es tu", "qui es-tu", "qui t'a", "qui t'a cree",
                  "qui t'a cree", "ton createur", "ton nom", "tu t'appelles",
                  "tu es qui", "que sais tu faire", "que peux tu faire",
                  "tu sais faire quoi")

# QUESTIONS META SUR L'IA (« es-tu une ia », « aussi puissant qu'un 8B ») :
# ni une tache de redaction ni un fait a verifier. Sans ce garde-fou, la
# longueur seule (>= 9 mots) declenchait le mode dissertation (plan +
# 3 sections + double passe = ~1400 tokens de CPU, soit ~150 s) sur une
# question fermee — d'ou la sortie hors-sujet repetee observee en prod.
# Le mode dissertation a ete supprime, mais ce garde-fou reste : il borne
# le BUDGET de tokens d'une question adressee au modele (110 tokens), ce qui
# n'a rien a voir avec le mode.
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
    # « intelligence artificielle » SEUL a ete retire : il faisait passer
    # pour une question SUR RontoGPT toute question dont l'IA est le SUJET
    # (« Compare l'intelligence artificielle et l'intelligence humaine »
    # recevait la reponse constante « je ne suis pas un 8B » au lieu d'une
    # analyse — bug reel trouve par scripts/banc_100.py). Les formulations
    # dirigees vers le modele restent couvertes plus haut : « es tu »,
    # « es-ce que tu es », « tu es une IA ».
    r"\b\d{1,3}[\s-]?b\b|\bgpt[-\s]?\d|\bgpt\b|\bchatgpt\b|\bopenai\b|"
    r"\bllama[-\s]?\d|\bmistral\b|\bdeepseek\b|\bclaude[-\s]?\d|\bgemini\b)",
    re.IGNORECASE)

# Reponse-constante pour les questions de capacite (0 s, zero hallucina-
# tion possible, meme esprit que _IDENTITE) : honnete et technique.
_CAPACITE = (
    "Je ne suis pas un modele 8B : le cerveau configure actuellement est "
    "Gemma-3-1B-it quantifie Q4_K_M, avec environ un milliard de parametres. "
    "RontoGPT combine ce modele local avec des solveurs exacts, un routeur, des "
    "sources de faits et des filtres. La plupart des traitements restent "
    "locaux; la recherche web et les APIs publiques necessitent une connexion. "
    "Un modele plus gros peut mieux traiter certaines nuances et raisonnements, "
    "mais demande davantage de RAM, de stockage et de temps. La qualite depend "
    "de la tache : le score 98/100 rapporte par le projet est celui d'un banc "
    "systeme de 100 questions, pas une mesure universelle. Le cerveau est "
    "configurable et peut aussi etre remplace par un modele 1,5B compatible.")


# ACCUSE DE RECEPTION (« ok », « merci », « d'accord ») : ni une question
# ni une tache, mais ce n'est PAS neutre pour autant. Sans garde-fou le
# 1B, privé de signal, se mettait à CONTINUER le format de tournure de la
# brique few-shot d'agents.py et rendait « R: Quel est le taux de change
# du euro a la date limite precedente ? » — en prod l'exemple du ciel
# était recopié et attribué a l'utilisateur (« Vous avez pose : Pourquoi
# le ciel est-il bleu ? »). Bug reel reproduit, 2026-10.
#
# PORTÉE MESURÉE : les SALUTATIONS sont volontairement EXCLUES. Avec la
# brique de style corrigée (cf. agents.py), le 1B répond parfaitement à
# « bonjour » -> « Bonjour ! Comment puis-je vous aider aujourd'hui ? ».
# Les intercepter dégraderait la réponse pour rien — et deux tests
# (test_aura, test_jev) utilisent « bonjour » comme question témoin.
_RE_BRIBE = re.compile(
    r"^(?:ok|okay|okey|d'?accord|entendu|compris|parfait|super|cool|"
    r"nickel|merci(?: beaucoup| bien)?|thanks|thx|"
    r"oui|non|ouais|yep|nope|exact|exactement|mdr|lol|ptdr|"
    r"ça marche|ca marche)[\s!.,;:?…]*$", re.IGNORECASE)

_BRIBE = ("Bien reçu. Dis-moi ton besoin — un calcul exact, un fait "
          "vérifié ou une explication — et je m'en occupe.")


def _est_bribe(question: str) -> bool:
    """Vrai si le message n'est QUE de la conversation, sans contenu.

    Ancre des deux bouts, 4 mots et 40 caracteres au maximum : une
    vraie demande courte (« combien font 2+2 ») ne tombe pas dedans.
    """
    q = (question or "").strip()
    if not q or len(q) > 40 or len(q.split()) > 4:
        return False
    return bool(_RE_BRIBE.match(q))


def _reponse_bribe(question: str) -> str | None:
    """Reponse constante pour un accuse de reception, sinon None."""
    return _BRIBE if _est_bribe(question) else None


# RECOPIE EXACTE : « ecris exactement le mot BANANE », « ecris la
# phrase : je teste RontoGPT ». Ce n'est PAS une tache de redaction — le 1B
# « corrigeait » la consigne au lieu de l'executer : au banc 100 il
# rendait « Je tests RontoGPT. » pour « je teste RontoGPT » et autre chose que
# BANANE. Une consigne de recopie se rend TELLE QUELLE, par le programme.
# Deux formes seulement (jamais de forme libre) pour eviter les faux
# positifs du genre « ecris la phrase correcte en anglais ».
_RE_RECOPIE = re.compile(
    r"\becris?\s+(?:moi\s+)?(?:"
    r"exactement\s+(?:le\s+mot|la\s+phrase|la\s+formule|le\s+texte)\s+(.+?)"
    r"|"
    r"(?:le\s+mot|la\s+phrase|la\s+formule|le\s+texte)\s*:\s*(.+?)"
    r")\s*$",
    re.IGNORECASE | re.DOTALL)


def _recopie(question: str) -> str | None:
    """Le texte a recopier tel quel, ou None si ce n'en est pas une."""
    m = _RE_RECOPIE.search((question or "").strip())
    if not m:
        return None
    texte = (m.group(1) or m.group(2) or "").strip()
    texte = texte.rstrip(" .,;:")      # ponctuation de la CONSIGNE
    if not texte or len(texte) > 300:
        return None
    return texte


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

# SOUS-ENSEMBLE STRICT de _RE_DEF, pour le FILET FINAL seulement.
#
# Deux demandes differentes etaient confondues dans `est_def` :
#   * « qu'est-ce que X / c'est quoi X / definis X » demande un FAIT
#     VERIFIABLE : sans ancrage, s'abstenir est la bonne reponse — inventer
#     une definition fausse avec aplomb est le defaut mesure sur ce projet ;
#   * « explique X / montre-moi X » demande un DEVELOPPEMENT : le modele
#     peut le produire de memoire, et s'abstenir y serait une non-reponse.
# `est_def` reste tel quel pour FORCER l'ancrage web (un contexte aide dans
# les deux cas). Seul le filet final distingue les deux.
#
# Consequence visible, et c'est ce qui a fait remonter le probleme : tant que
# le mode dissertation existait, toute question contenant « explique »
# partait en redaction AVANT d'atteindre le filet — la confusion etait sans
# effet. Sa suppression l'a rendue visible : hors-ligne, « Explique la
# photosynthese » repondait « je ne peux pas verifier cette definition ».
_RE_DEF_LOOKUP = re.compile(
    r"\b(qu'?est.?ce.?que|qu'?est.?ce.?qu'|c'?est.?quoi|definis|"
    r"qu'elle? est la definition|la definition de|signifie)\b", re.IGNORECASE)


# ── Pertinence d'une source pour une definition (routage HYBRIDE) ────────
# Une definition a DEUX regimes selon que la source parle vraiment du sujet :
#   * source FORTE (le sujet y figure) -> synthese ancree par le 1B ;
#   * source FAIBLE / hors-sujet       -> citation telle quelle (anti-
#     hallucination), car un 1B laisse seul invente : mesure du 8 oct. 2026,
#     sur « c'est quoi une fonction canonique », il a rendu « si x = y alors
#     f(x) = f(y) » — la definition d'une fonction BIEN definie, fausse ici.
# Le signal de pertinence est volontairement STRICT : le sujet doit apparaitre
# en clair (phrase contigue) dans la source. « fonction canonique » ne figure
# pas dans des extraits parlant de « forme canonique » ou de « surjection
# canonique » : on cite donc au lieu de broder.
def _sans_accents(txt: str) -> str:
    import unicodedata
    n = unicodedata.normalize("NFKD", txt.lower())
    return "".join(c for c in n if not unicodedata.combining(c))


_AMORCE_DEF = re.compile(
    r"^\W*(?:qu'?est[\s-]?ce[\s-]?qu(?:e|')|c'?est[\s-]?quoi|"
    r"explique(?:[\s-]?moi)?|definis|montre[\s-]?moi|que\s+signifie|"
    r"definition\s+de)\s*", re.IGNORECASE)
_ARTICLE_DEF = re.compile(r"^(?:une?|le|la|les|des|du|de\s+la|d'|l')\s+",
                          re.IGNORECASE)


def _sujet_definition(question: str) -> str:
    """Le sujet nu d'une question de definition (« fonction canonique »)."""
    sujet = _AMORCE_DEF.sub("", _sans_accents(question or "")).strip()
    sujet = _ARTICLE_DEF.sub("", sujet).strip()
    sujet = re.sub(r"[?!.;,]+$", "", sujet).strip()
    return sujet


def _sans_fragment_incomplet(texte: str) -> str:
    """Retire les fragments finaux en pleine phrase (snippets tronques).

    Un moteur de recherche renvoie des extraits coupes net, souvent juste
    avant une formule (« ... une factorisation par a : »). Les citer tels
    quels DONNE L'IMPRESSION que la reponse s'arrete au milieu. On recule
    donc au dernier point de fin de phrase : le texte rendu est complet.
    """
    lignes: list[str] = []
    for ligne in (texte or "").splitlines():
        t = ligne.rstrip()
        if not t:
            continue
        if t[-1] in ".!?…»\"":
            lignes.append(t)
            continue
        fins = list(re.finditer(r"[.!?…](?=\s)", t))
        lignes.append(t[:fins[-1].end()].rstrip() if fins else t)
    return "\n".join(lignes)


def _source_pertinente(question: str, contexte_web: str) -> bool:
    """La source parle-t-elle VRAIMENT du sujet demande ?

    Strict a dessein : le sujet doit apparaitre comme phrase contigue. Un
    simple recouvrement de mots (« fonction » et « canonique » presents
    separement, dans un autre sens) ne suffit pas — c'est exactement le cas
    qui doit rester en citation.
    """
    sujet = _sujet_definition(question)
    if len(sujet) < 3 or not contexte_web:
        return False
    return sujet in _sans_accents(contexte_web)

from . import expert_symbolique, memoire_web, autoamelioration, filtre_instantane, rag_binaire
from . import registre_experts, chaines_numeriques, clarification
from . import problemes_arithmetiques
from . import micro_faits
from . import llama_cerveau, raisonneur, graphe_faits
from . import logique as solveur_logique
from . import potcode
from . import arbre_reflexion
from . import apis_publiques
from . import agents
from . import adaptateurs
from . import calcul_achat
from . import calcul_verbal
from . import calcul_taux
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
#   question ou le 1B hallucine. (StyleGuard garantit deja
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

# Problemes de TAUX / arithmetique en mots (escargot qui grimpe, robinet qui
# remplit, train qui rattrape) : le 1B y pose une equation fausse. Signature
# volontairement ETROITE — un SUJET mobile/quantitatif ET une phrase de
# mouvement ou de cadence. Un mot seul ferait basculer des questions
# ordinaires dans une passe de reflexion qui n'a rien a y faire.
_SUJET_MOUVEMENT = re.compile(
    r"\b(escargot|limace|grenouille|fourmi|chenille|robinet|fontaine|"
    r"citerne|reservoir|cuve|piscine|bateau|nageur|train|voiture|"
    r"coureur|cycliste)\b")
_PHRASE_TAUX = re.compile(
    r"\b(monte|grimpe|glisse|redescend|remonte|se remplit|se vide|"
    r"avance|recule|par (jour|heure|nuit|minute)|"
    r"chaque (jour|heure|nuit|minute))\b")
#: Prompt systeme MINIMAL d'une question de RAISONNEMENT (puzzle/enigme,
#: think=True). Le prompt complet porte des CONSIGNES D'OUTIL (« Pour
#: lire un fichier texte... », « cherche dans l'historique... ») : le 1B
#: les RECOPIE au lieu de raisonner (bug reel 10 oct. 2026 : l'enigme
#: « escargot » rendait « Pour lire un fichier texte de l'ordinateur... »).
#: Ici, aucune consigne copiable : la tache, et rien d'autre.
_SYSTEME_RAISONNEMENT = (
    "Resous UNIQUEMENT le probleme pose, en francais. "
    "Ecris le calcul puis le resultat, en trois phrases maximum. "
    "N'ecris aucune consigne, aucun exemple, aucun plan : reponds "
    "seulement au probleme.")


def _sans_accents(texte: str) -> str:
    """Minuscules sans diacritiques (« Redige » -> « redige »).

    Meme convention que calcul_achat / calcul_taux : le detecteur doit
    reconnaitre la question accentuee comme la version sans accents (le
    banc ecrit ses questions sans diacritiques, la GUI les ecrit avec).
    """
    import unicodedata
    n = unicodedata.normalize("NFKD", (texte or "").lower())
    return "".join(c for c in n if not unicodedata.combining(c))


# DEMANDE DE DEVELOPPEMENT : la question veut un TEXTE suivi, pas un fait.
# Pourquoi une liste separee de _RE_DEF ? Parce que le REMEDE est different :
# _RE_DEF sert a FORCER l'ancrage web, celui-ci a RETIRER du prompt systeme
# tout ce qui se recopie (cf. _SYSTEME_REDACTION). Volontairement etroit :
# « explique » seul n'y est pas (« explique la photosynthese » reste un
# lookup ancrable) — il faut « pourquoi », ou un mot de redaction explicite.
_RE_REDACTION = re.compile(
    r"(^|[^a-z])("
    r"redige|rediger|redaction|dissertation|disserte|essai|"
    r"argumente|argumentaire|argumentation|developpe|developpement|"
    r"paragraphe|"
    r"avantages et les inconvenients|avantages et inconvenients|"
    r"en trois parties|en plusieurs parties|"
    r"analyse|analyser|comparaison|compare|comparer|"
    r"explique pourquoi"
    r")([^a-z]|$)", re.IGNORECASE)


def _est_redaction(question: str) -> bool:
    """Vrai si la question demande une REDACTION longue (developpement)."""
    return bool(_RE_REDACTION.search(_sans_accents(question)))


#: Prompt systeme MINIMAL d'une demande de REDACTION. Meme cause, meme
#: remede que _SYSTEME_RAISONNEMENT ci-dessus (et meme symptome : le 1B
#: recopie le prompt au lieu d'y repondre).
#:
#: Mesure du banc maison (11 oct. 2026, categorie « riche », 6 questions
#: sur 10) : avec le prompt complet, « Analyse les avantages et les
#: inconvenients du teletravail » et « Compare la democratie athenienne et
#: la democratie moderne » rendaient l'une le bloc LEXIQUE recopie, l'autre
#: « [RontoGPT] Reflexion interrompue » ; toutes deux identiques d'une question
#: a l'autre (911 car). Avec ce prompt, les deux memes questions rendent
#: 1 423 et 1 412 car de redaction SUR LE SUJET.
#: Aucun plan n'est impose : le mode « dissertation » multi-pass a ete
#: supprime — on ne fait ici que retirer ce qui se recopie.
_SYSTEME_REDACTION = (
    "Redige en francais une reponse developpee et structuree en plusieurs "
    "paragraphes, qui traite la question posee. N'ecris aucune consigne, "
    "aucun exemple, aucun titre de section, aucun libelle : redige "
    "seulement la reponse, sans la commenter.")

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


class RontoGPT:
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
        # Vrai si un expert exact (logique, code, calcul, fast-lang) a deja
        # tranche : lu par _generer() pour le filet anti-hallucination.
        # Initialise ici : sans ca, appeler _generer() hors du pipeline
        # complet (tests, scripts) levait AttributeError.
        self._special_traite = False
        # ETAT DE LA PORTE DE CLARIFICATION (aura/clarification.py) : la
        # relance en attente vit ICI, cote instance — un seul cerveau par
        # process (API, GUI), donc un seul suivi de conversation possible.
        # Un seul tour : des que la precision est recollee, l'etat est vide.
        self._clarif = clarification.Suivi()
        # Contexte API publique recolte par _executer_experts (consomme par
        # le pipeline APRES le compresseur : la ligne de donnees ne doit
        # pas partir au top-3 des phrases « utiles »).
        self._contexte_api = ""
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
        # une relance de clarification appartient a la conversation qu'on
        # vient de quitter : la garder fusionnerait une precision avec une
        # question d'un autre sujet.
        self._clarif.annuler()

    # -- routage intelligent ------------------------------------------------

    def analyser(self, question: str) -> dict:
        try:
            return self._routeur.classer(question)
        except Exception as e:
            LOG.warning("[routeur] fallback (%s)", e)
            return self._routeur.routeur_classique(question)

    # -- decision en un cycle (idea JEV #3) ---------------------------------

    def _decider(self, question: str):
        """Un cycle de decision : niveau 0 exact, puis routeur si necessaire.

        Le niveau 0 repond en moins d'une milliseconde. Le lancer en parallele
        du routeur faisait perdre ce travail sur chaque hit exact : le thread
        de classification continuait en arriere-plan meme si son resultat
        etait ignore. On ne paie le routage que pour les questions qui en ont
        besoin ; son surcout est negligeable face a une generation LLM.
        Renvoie (instant, analyse) — analyse vaut None si le niveau 0 a
        tranche tout seul.

        `filtre_instantane.repondre` est ici l'etage EXACT seul. Le cache
        semantique n'est PLUS consulte ici : il l'est plus loin, apres tous
        les solveurs exacts (cf. le bloc « CACHE SEMANTIQUE » dans
        executer_detaille). C'est une contrainte d'ordre, pas une
        optimisation : un approximatif ne doit pas primer sur un exact.
        """
        instant = filtre_instantane.repondre(question)
        if instant is not None:
            return instant, None
        return None, self.analyser(question)

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
        if "web" in experts:
            taches["web"] = (memoire_web.chercher, (question,))
        if "math" in experts:
            taches["pgs"] = (self.resoudre_numerique, (X, y))
        # API PUBLIQUES (liste public-apis) : meteo, devises, definitions,
        # livres, geocodage — sans cle, stdlib. La detection est une regex
        # LOCALE (zero octet si hors sujet) et la tache part en parallele :
        # sa latence se cache sous celle de DuckDuckGo.
        if apis_publiques.detecte(question):
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
        redaction = bool(not think and _est_redaction(question))
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
            except Exception:
                pass
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
        return self._llama(question, contexte_web, formule,
                           historique=historique_generation,
                           contexte_faits=contexte_faits,
                           systeme=systeme_envoye,
                           categorie=categorie,
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
        q = question.lower()
        return (RontoGPT._est_puzzle(question)
                or bool(_ENIGME_LOGIQUE.search(q))
                # probleme de taux : les DEUX signatures, pas l'une ou l'autre
                or bool(_SUJET_MOUVEMENT.search(q)
                        and _PHRASE_TAUX.search(q)))

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
            return not ctx.est_puzzle
        if nom == "cache-semantique":
            return not ctx.a_donnees
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
            if _est_refus(memo):
                LOG.info("[niveau0] refus servi par le cache -> pipeline")
                return None
            return {"question": question, "experts": experts | {"memoire"},
                    "analyse": analyse, "cerveau_choisi": nom,
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": memo}
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

    def _consommer_relance(self, question: str) -> tuple[str, bool]:
        """Recolle la precision du tour precedent, s'il y en avait une.

        Renvoie (question, relance). `relance` vrai = une porte a deja ete
        posee pour ce tour : la porte est alors COURT-CIRCUITEE, ce qui
        garantit « un seul tour » (l'utilisateur ne peut pas se voir poser
        une seconde serie de questions, meme s'il repond vaguement).
        """
        if not self._clarif.en_attente():
            return question, False
        if _est_bribe(question):
            # « ok » n'apporte AUCUNE precision : recoller produirait
            # « Explique ok ». On referme la relance et on traite le tour
            # normalement (jamais de conversation bloquee en attente).
            LOG.info("[clarification] acquittement sans precision -> "
                     "relance abandonnee")
            self._clarif.annuler()
            return question, False
        fusion = self._clarif.consommer(question)
        if not fusion:
            return question, False
        LOG.info("[clarification] precision recollee : %s", fusion[:120])
        return fusion, True

    def _porte_clarification(self, question: str, X, y) -> dict | None:
        """Porte : question trop ouverte -> questions de relance, pas de reponse."""
        if not clarification.actif():
            return None
        if X is not None and y is not None:
            return None                 # calcul numerique demande : rien a preciser
        besoin = clarification.detecter(
            question,
            a_contexte=clarification.a_un_sujet(self._historique.fenetre()))
        if besoin is None:
            return None
        self._clarif.poser(question, besoin)
        LOG.info("[clarification] %s (%s) : %s", besoin.famille,
                 besoin.raison, " | ".join(besoin.questions))
        reponse = besoin.texte()
        # La relance est un TOUR de conversation a part entiere : le cerveau
        # doit la voir pour ne pas repondre deux fois a la meme question.
        # En revanche elle n'est JAMAIS mise au cache des reponses : ce
        # n'est pas une reponse a la question posee.
        self._historique.ajouter("user", question)
        self._historique.ajouter("assistant", reponse)
        return {"question": question, "experts": {"clarification"},
                "analyse": {"methodes": ["clarification"]},
                "cerveau_choisi": "clarification",
                "contexte_web": "", "formule": "", "erreur_pgs": None,
                "clarification": {"famille": besoin.famille,
                                  "manque": list(besoin.manque),
                                  "questions": list(besoin.questions),
                                  "raison": besoin.raison},
                "en_attente": True,
                "reponse": reponse}

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
        # ECHEANCE ABSOLUE (et non un simple drapeau) : elle est POSEE pour
        # ce fil et lue par la generation, qui la convertit en budget de
        # tokens. Sans ca, AURA_BUDGET_S ne bornait AUCUNE generation :
        # 136 s observees au banc pour un budget annonce de 45 s.
        # Passe par un ContextVar (llama_cerveau.poser_echeance) et non par
        # un kwarg : generer() est appele aussi par les agents, le PoT et
        # l'API, qui n'ont pas a connaitre cette valeur de contexte.
        deadline = (t0 + _BUDGET_S) if _BUDGET_S > 0 else None
        llama_cerveau.poser_echeance(deadline)
        try:
            return self._executer_detaille(question, X, y, t0)
        finally:
            # Nettoyage obligatoire : les workers de l'API reutilisent leur fil.
            llama_cerveau.poser_echeance(None)

    def _executer_detaille(self, question: str, X=None, y=None,
                           t0: float | None = None) -> dict:
        t0 = t0 if t0 is not None else time.time()
        # ── RELANCE DE CLARIFICATION EN ATTENTE (un seul tour) ─────────
        # Si la porte a demande une precision au tour precedent, ce message
        # EST la precision : on la recolle a la question d'origine et le
        # pipeline repart sur une question complete. C'est la seule chose
        # qui est faite avant les briques deterministes : sans ce recollage,
        # « Convertis 100 » + « en km » ne redeviendrait jamais la phrase que
        # `pal.py` sait calculer.
        question, relance = self._consommer_relance(question)
        # ── BRIBES CONVERSATIONNELLES (« ok », « merci », « salut ») ────
        # Ce n'est PAS une question : on repond par une constante, sans
        # jamais toucher au LLM (zero hallucination, zero cout).
        bribe = _reponse_bribe(question)
        if bribe is not None:
            return {"question": question, "experts": {"conversation"},
                    "analyse": {"methodes": ["bribe"]},
                    "cerveau_choisi": "constante",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": bribe}
        # RECOPIE EXACTE : consigne deterministe, jamais confiee au modele.
        recopie = _recopie(question)
        if recopie is not None:
            return {"question": question, "experts": {"recopie"},
                    "analyse": {"methodes": ["recopie"]},
                    "cerveau_choisi": "constante",
                    "contexte_web": "", "formule": "",
                    "erreur_pgs": None, "reponse": recopie}
        # ── CYCLE UNIQUE (idea JEV #3) ─────────────────────────────────
        # maths directes + routage en UN passage : la majorite des questions
        # quotidiennes n'ont PAS besoin du LLM. NB : depuis le 6 oct. 2026
        # cet etage est EXACT — le cache semantique a ete deplace apres les
        # solveurs (cf. plus bas), pour qu'une memoire approximative ne
        # puisse plus repondre avant un calcul juste.
        instant, analyse = self._decider(question)
        if instant is not None and not (X and y):
            # NETTOYAGE : le calcul direct peut rendre une chaine brute a
            # nettoyer (doublon, libelle de plan recopie).
            instant = llama_cerveau._sans_doublons(instant)
            instant = llama_cerveau._sans_libelles(instant) or instant
            if _est_refus(instant):
                LOG.info("[niveau0] refus servi -> pipeline complet")
                instant, analyse = None, self.analyser(question)
            else:
                return {"question": question, "experts": {"instantane"},
                        "analyse": {"methodes": ["niveau0"]},
                        "cerveau_choisi": "niveau0-instantane",
                        "contexte_web": "", "formule": "",
                        "erreur_pgs": None, "reponse": instant}

        # IDENTITE : reponse constante, hors des poids — l'apparence d'RontoGPT
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
        # meta OUVERTE (longue) : elle repond, mais le budget de tokens est
        # borne (reponse courte, pas 1400 tokens de CPU).
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
        # Filet final : uniquement les LOOKUPS (cf. _RE_DEF_LOOKUP).
        est_def_lookup = bool(_RE_DEF_LOOKUP.search(question)
                              or _RE_PHRASE_MIN.search(question))
        if est_def:
            experts.discard("general")
            experts.add("web")
        # DETECTION UNIQUE (executes une seule fois) :
        # - est_puzzle pilote le gate RAG et la passe PoT ;
        # - auto_think pilote le brouillon <thinking> a la generation
        #   finale (les puzzles/enigmes qui ont RATE leurs experts
        #   deterministes repartent avec une passe de reflexion masquee).
        est_puzzle = self._est_puzzle(question)
        auto_think = self._auto_think(question)

        # ── EXPERTS, DANS L'ORDRE DECLARE (aura/registre_experts.py) ────
        # Ce bloc etait une SUITE DE `if` : l'ordre n'existait que dans
        # l'ordre des lignes, donc personne ne pouvait le relire — et c'est
        # ainsi que le cache semantique avait fini AVANT les solveurs
        # exacts. L'ordre est desormais declare (avec sa raison) et verifie
        # par tests/test_registre_experts.py.
        #
        # NE PAS re-inserer d'expert en `if` ici : ajouter une entree au
        # registre, son appel dans `_appel_expert` et sa garde dans
        # `_expert_applicable` — le test refuse un rang incoherent ou un
        # expert non branche.
        #
        # Les solveurs exacts n'ont AUCUNE condition : un probleme raconte
        # peut etre long, il reste un CALCUL — la longueur ne doit pas le
        # priver de l'exactitude.
        contexte = registre_experts.Contexte(
            question=question, est_puzzle=est_puzzle,
            a_donnees=bool(X and y))
        self._special_traite = False
        for nom_expert in registre_experts.noms():
            if not self._expert_applicable(nom_expert, contexte):
                continue
            reponse_expert = self._appel_expert(nom_expert, question, X, y,
                                                t0)
            if reponse_expert is None:
                continue
            resultat = self._resultat_expert(nom_expert, reponse_expert,
                                             question, analyse, experts)
            if resultat is None:        # hit refuse (ex : refus memorise)
                continue
            self._special_traite = True
            return resultat

        # ── PORTE DE CLARIFICATION (cf. registre_experts.PORTES) ───────
        # Question trop ouverte (« Explique » sans objet, « Ecris un poeme »
        # sans sujet, « Convertis 100 » sans unite cible) : plutot que de
        # laisser le 1B inventer un sujet, la porte pose 1 a 3 questions et
        # ne repond pas. APRES les experts (une question exacte doit rester
        # exacte), AVANT la generation (demander coute moins cher qu'une
        # reponse fausse). Court-circuitee quand une precision vient d'etre
        # recollee : un seul tour, jamais de ping-pong.
        if not relance:
            resultat_clarif = self._porte_clarification(question, X, y)
            if resultat_clarif is not None:
                return resultat_clarif

        # FAN-OUT PARALLELE (idea JEV #1) : web ∥ PGS ∥ API publiques, un
        # aller-retour
        contexte_web, formule = self._executer_experts(
            experts, question, X, y)
        # Conserver la provenance avant compression du contexte web.
        sources_web = memoire_web.extraire_sources(contexte_web)
        # API PUBLIQUES (meteo, devises, definitions, livres, geoloc) :
        # recollee APRES le compresseur — une ligne de donnees structurees
        # ne doit pas etre jetee au top-3 — puis livree telle quelle si
        # elle est SEULE : zero generation (rapide), zero hallucination,
        # et le 1B ne peut plus refuser des donnees « en temps reel »
        # (refus observe en direct alors que la meteo etait au contexte).
        api = self._contexte_api
        self._contexte_api = ""
        sans_web = not contexte_web
        # AGENT 2 (compresseur RAG) : le contexte web brut est filtre —
        # seules les phrases utiles a la question alimentent le LLM
        if contexte_web:
            try:
                contexte_web = agents.compresser(contexte_web, question)
            except Exception as e:
                LOG.info("[agents] compression impossible (%s) -> texte brut", e)
        if api:
            contexte_web = (f"{contexte_web}\n{api}"
                            if contexte_web else api)
        if api and sans_web and not formule:
            LOG.info("[api] livraison extractive : %d car.", len(api))
            filtre_instantane.enregistrer(question, api)
            return {"question": question, "experts": experts | {"api"},
                    "analyse": analyse, "cerveau_choisi": "extraction-api",
                    "contexte_web": api, "formule": "",
                    "erreur_pgs": None, "reponse": api}
        if contexte_web:
            # DEFINITIONS : routage HYBRIDE selon la pertinence de la source.
            #
            # 1) Source FAIBLE (le sujet n'y figure pas en clair) -> ancrage
            #    extractif : on CITE le contexte verifie au lieu de le laisser
            #    reformuler par le 1B, qui invente avec aplomb (mesure du
            #    8 oct. 2026 : « fonction canonique » -> « si x = y alors
            #    f(x) = f(y) », faux). Extraction = zero generation = zero
            #    hallucination possible.
            # 2) Source FORTE (le sujet y figure) -> synthese ancree : le 1B
            #    redige a partir du VERIFIE. Lisible et, source claire aidant,
            #    fiable. Une definition reste un lookup, jamais une redaction
            #    libre.
            # GATE = `est_def_lookup`, PAS `est_def` : « explique X / montre-moi X »
            # demande un DEVELOPPEMENT (cf. _RE_DEF_LOOKUP) — le faire passer par
            # l'ancrage extractif le livrait comme un LOOKUP sans source. Mesure du
            # banc du 11 oct. 2026 : « Explique pourquoi les oceans sont essentiels a
            # l'equilibre climatique » rendait « Aucune source ne definit directement
            # « ... » » — une non-reponse a la place d'un developpement. `est_def`
            # reste vrai et continue de FORCER le web (l'ancrage aide dans les deux
            # cas) ; seul le regime d'EXTRACTION est reserve aux lookups.
            if not formule and est_def_lookup:
                if _source_pertinente(question, contexte_web):
                    LOG.info("[def] source pertinente -> synthese ancree")
                    rep = self._generer(question, contexte_web, "",
                                        personnalite=None, categorie="web",
                                        est_def=est_def,
                                        est_def_lookup=est_def_lookup)
                    rep = llama_cerveau._sans_doublons(rep) or rep
                    rep = memoire_web.ajouter_citations(rep, contexte_web)
                    filtre_instantane.enregistrer(question, rep)
                    return {"question": question, "experts": experts | {"web"},
                            "analyse": analyse, "cerveau_choisi": "def-web+1b",
                            "contexte_web": contexte_web, "formule": "",
                            "erreur_pgs": None, "reponse": rep}
                # dedup : deux extraits de recherche identiques ne doivent
                # pas etre livres cote a cote (phrase repetee a l'ecran).
                # Honnetete : quand le sujet ne figure nulle part, on le DIT
                # avant de citer — sinon l'utilisateur croit a une reponse.
                _sujet = _sujet_definition(question)
                amorce = (f"Aucune source ne definit directement « {_sujet} ». "
                          if len(_sujet) >= 3 else "")
                reponse_extraite = (
                    f"{amorce}D'apres les sources verifiees en ligne :\n"
                    + llama_cerveau._sans_doublons(
                        _sans_fragment_incomplet(contexte_web)))
                reponse_extraite = memoire_web.ajouter_citations(
                    reponse_extraite, contexte_web)
                filtre_instantane.enregistrer(question, reponse_extraite)
                return {"question": question, "experts": experts | {"web"},
                        "analyse": analyse, "cerveau_choisi": "extraction-web",
                        "contexte_web": contexte_web, "formule": formule,
                        "erreur_pgs": None, "reponse": reponse_extraite}
            # AGENT 2b (hyper-compression) : phrases -> triplets semantiques
            # (sujet | relation | objet) — protege la fenetre du 1B et
            # nourrit le graphe de faits au passage
            try:
                contexte_web = agents.en_triplets(contexte_web, question)
            except Exception as e:
                LOG.info("[agents] triplets impossible (%s) -> texte compresse", e)
            liens = [f"Titre : {titre} | Source : {url}"
                     for titre, url in sources_web]
            if liens:
                contexte_web += "\nSOURCES WEB :\n" + "\n".join(liens)

        # AGENT 1 (planificateur) : les taches actionables complexes sont
        # decoupees en 2-3 etapes AVANT la redaction
        plan = ""
        if not formule:
            try:
                plan = agents.planifier(question) or ""
            except Exception as e:
                LOG.info("[agents] planification impossible (%s) -> sans plan", e)
            if plan:
                LOG.info("[agents] plan insere : %d lignes", len(plan.splitlines()))

        # RAISONNEMENT PoT (Program-of-Thoughts) : les puzzles d'ages/
        # relations sans donnees numeriques vont au raisonneur — le 1B
        # ecrit les etapes, l'AST les verifie exactement. Repli auto.
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
        # meta-ia ouverte : budget borne (reponse courte, jamais 1400 tokens)
        # AUTO-THINK : puzzles, enigmes et problemes en mots seulement — et
        # seulement s'il reste du budget.
        _budget_ok = _BUDGET_S <= 0 or time.time() - t0 < _BUDGET_S
        _use_think = auto_think and _budget_ok
        reponse = self._generer(question_envoyee, contexte_web, formule,
                                 contexte_faits=contexte_faits,
                                 personnalite=personnalite,
                                 categorie=categorie, est_def=est_def,
                                 est_def_lookup=est_def_lookup,
                                 max_tokens=(110 if meta_ia else None),
                                 think=_use_think)
        # REPLI SANS-REFLEXION : la reflexion coute 2x de budget et le 1B
        # peut ne jamais la fermer -> le cerveau renvoie un message
        # d'interruption. Sur un probleme de taux, le chemin DIRECT repond
        # souvent (mal, mais utilisable) : rendre ce message serait une
        # regression. On retente UNE fois sans reflexion, et seulement si
        # l'echeance laisse le temps d'une generation.
        if (_use_think
                and reponse.startswith("[RontoGPT] Reflexion interrompue")
                and (_BUDGET_S <= 0 or time.time() - t0 < _BUDGET_S)):
            LOG.info("[auto-think] reflexion non terminee -> repli sans "
                     "reflexion")
            reponse = self._generer(question_envoyee, contexte_web, formule,
                                     contexte_faits=contexte_faits,
                                     personnalite=personnalite,
                                     categorie=categorie, est_def=est_def,
                                     est_def_lookup=est_def_lookup,
                                     max_tokens=(110 if meta_ia else None),
                                     think=False)
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
            # LIBELLES DE CONSIGNE : « SECTION A REDIGER : PARTIE n »
            # recopie par le 1B ne doit JAMAIS arriver a l'utilisateur
            # (observe au banc sur cette balise ecolee).
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
        # VERIFICATION OUTILLEE (pattern CRITIC) : preuve web avant livraison
        verifiee = False
        if ((_BUDGET_S <= 0 or time.time() - t0 < _BUDGET_S)
                and self._a_besoin_verification(question, contexte_web,
                                                reponse)):
            avant_verification = reponse
            reponse = self._verifier_au_web(question, reponse)
            # N'apprendre que si la reponse revisee a change et porte au
            # moins deux liens distincts. Un simple extrait ou un seul site
            # ne suffit pas a consolider un fait dans la memoire longue.
            verifiee = (
                reponse.strip() != avant_verification.strip()
                and len(memoire_web.extraire_sources(reponse)) >= 2)
            # RECONSOLIDATION : la reponse verifiee remplace l'ancienne au
            # cache (sinon une reponse fausse d'avant reste collée a vie)
            if verifiee:
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
        reponse = memoire_web.ajouter_citations(reponse, contexte_web)
        # memorise pour les futures questions (cache semantique + conversation)
        filtre_instantane.enregistrer(question, reponse)
        self._historique.ajouter("user", question)
        self._historique.ajouter("assistant", reponse)
        # PENSEE ET AUTO-CORRECTION DU TOUR : elles sont jointes au resultat
        # EXPLICITEMENT, sous leurs propres cles. C'est le pendant cote
        # donnees de la separation des flux : `reponse` ne contient QUE la
        # reponse, et qui veut la reflexion doit demander `pensee`. Un
        # consommateur qui ignore ces cles (cache, ancrage, graphe) ne voit
        # donc jamais la reflexion du modele passer pour une reponse.
        return {"question": question, "experts": experts, "analyse": analyse,
                "cerveau_choisi": self._nom_cerveau(), "contexte_web": contexte_web,
                "formule": formule, "erreur_pgs": self.derniere_erreur,
                "pensee": llama_cerveau.reflexion_du_tour(),
                "auto_correction": llama_cerveau.alerte_du_tour(),
                "reponse": reponse}

    @staticmethod
    def _nom_cerveau() -> str:
        """Identifiant du cerveau REELLEMENT charge, pour le diagnostic.

        La valeur etait ecrite en dur (« llama-3.2-1b ») : au moment du
        passage a Gemma, toute la chaine de diagnostic aurait continue
        d'annoncer Llama — un journal qui ment sur le modele charge coute
        plus cher que pas de journal. La famille est desormais LUE dans
        l'en-tete du GGUF (aura/gguf_meta.py), pas deduite d'un nom de
        fichier ni d'une constante.
        """
        return f"{llama_cerveau.famille() or 'inconnu'}-1b"


# Ancien nom public conservé pour les extensions et scripts existants.
Aura1B = RontoGPT
