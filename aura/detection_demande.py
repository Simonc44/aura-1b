"""Detection de la demande (verification, enigmes, redaction, code) et
 prompts systeme associes. Extrait de aura/orchestrateur.py ; aucun
 changement de comportement."""
import re
from . import raisonneur



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
    r"dit toujours vrai|dit toujours la verite)\b|"
    # Attribution / ordonnancement / contraintes sans chiffres :
    # « Pierre, Jeanne et Loic occupent trois postes... »,
    # « qui fait quoi / qui habite ou / qui a quel poste ».
    # FIX (12 oct. 2026) : le routeur flou ne basculait pas sur le SAT pour
    # les enigmes d'attribution sans mots-cles chevalier/menteur — le 1B
    # recevait la consigne brute et bégayait/reformulait.
    r"\b(occupent|occupe|poste|role|profession|metier|couleur|voisin|"
    r"maison|classement|rang|place|ordre|attribution|affectation|"
    r"qui est le|qui occupe|qui fait|qui habite|qui possede|qui aime)\b"
    r".{0,80}\b(parmi|entre|les trois|les deux|chacun|chacune|"
    r"l'un d'eux|l'une d'elles|differents?|distincts?)\b",
    re.IGNORECASE)

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
_RE_SORTIE_LONGUE = re.compile(
    r"\b(?:une|un|[1-9]\d*)\s+pages?\b|\b\d{2,4}\s+mots?\b|"
    r"\bplusieurs paragraphes\b|\breponse tres detaillee\b",
    re.IGNORECASE)


def _est_redaction(question: str) -> bool:
    """Vrai si la question demande une REDACTION longue (developpement)."""
    return bool(_RE_REDACTION.search(_sans_accents(question)))


# DEMANDE D'ECRITURE (production d'un texte) : la question ne demande PAS un
# fait a retrouver mais un texte a PRODUIRE. Cette distinction manquait, et
# elle avait une consequence visible : le mot « histoire de » figure dans
# _RE_PHRASE_MIN (pense pour « l'histoire de la Revolution »), donc « Ecris
# une courte histoire de trois phrases » etait classe LOOKUP -> web force ->
# regime extractif -> « Aucune source ne definit directement « ecris une
# courte histoire... » ». Mesure du 11 oct. 2026 : une NON-REPONSE la ou
# l'utilisateur attendait une histoire.
# Le motif est volontairement ETROIT : un VERBE de production ET un OBJET
# textuel proches. « Ecris exactement le mot banane » reste un lookup (test
# dedie dans test_routage_qualite), et « redige une dissertation » continue
# de passer par _RE_REDACTION.
_RE_ECRITURE = re.compile(
    r"\b(ecris|ecrire|raconte|raconter|invente|inventer|imagine|imaginer|"
    r"compose|composer|genere|generer)\b[^.!?]{0,40}?"
    r"\b(histoire|conte|recit|poeme|poesie|chanson|lettre|texte|dialogue|"
    r"scene|scenario|paragraphe|description|portrait|fable|nouvelle|roman|"
    r"page|article|rapport)\b")


def _est_demande_ecriture(question: str) -> bool:
    # Vrai si la question demande de PRODUIRE un texte.
    return bool(_RE_ECRITURE.search(_sans_accents(question)))


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
    r"\b(ecris|ecrire|donne|fournis|code|coder|programme|programmer|implemente|"
    r"implementer|genere|generer|cree|creer|construis|construire|"
    r"developpe|developper|write|implement|generate|create|build)\b"
    r"[^.?!]{0,100}\b(du code|code|python|javascript|typescript|html|css|"
    r"script|programme|program|application|app|fonction|function|classe|"
    r"class|module|composant|component|algorithme|algorithm|site web|"
    r"website|api|requete sql|sql query)\b|"
    r"\b(code|script|fonction|function|programme|program|application|app|"
    r"classe|class|module|composant|component)\b.{0,50}\b"
    r"(python|javascript|typescript|html|css|sql)\b", re.IGNORECASE)

# Le solveur de code sandboxe impose `resoudre(...)`, sans imports. On ne
# l'utilise que pour une demande d'algorithme/fonction Python que ce contrat
# peut réellement représenter. Les autres demandes vont directement au LLM.
_DEMANDE_CODE_SANDBOX = re.compile(
    r"\b(fonction|algorithme)\b.{0,50}\b(python)\b|"
    r"\b(python)\b.{0,50}\b(fonction|algorithme)\b", re.IGNORECASE)

# VERROUILLAGE CODE : si la requete contient des marqueurs evidents de
# programmation, l'expert web est DESACTIVE categoriquement — empeche le
# crash d'aiguillage ou une demande de script bascule sur DuckDuckGo.
# FIX (12 oct. 2026) : « ecris une fonction Python » partait en recherche
# internet si la question contenait un mot-cle temporel (ex: « Python 3.12 »).
_VERROUILLAGE_CODE = re.compile(
    r"\b(ecris?|ecrire|redige|cree?|implementer?|code|programme)"
    r"\b.{0,80}"
    r"\b(def |fonction|function|assert|class |import|python|javascript|"
    r"typescript|script|algorithme|sql|html|css)\b|"
    r"\b(def |assert |class )",  # mots-cles syntaxiques Python directs
    re.IGNORECASE)


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
