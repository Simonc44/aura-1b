"""Les 5 agents cognitifs d'Aura : petite tete, grands gestes.

Chaque agent est une FONCTION pure Python (pas un LLM) qui cadre le
travail du cerveau 1B — le meme coup que PoT/solveur : le modele propose,
le programme structure. Zero dependance, zero appel reseau.

1. Planificateur      : les taches actionables sont decoupees en 2-3 etapes
                        AVANT la redaction (le 1B ne s'effondre pas face a
                        un « organise-moi un voyage »).
2. Compresseur RAG    : le contexte web brut est filtre — 3 phrases les
                        plus proches de la question (la memoire courte du
                        1B n'est pas etouffee par le bruit d'internet).
3. Rédacteur          : les tics de langage typiques du 1B sont retires
                        apres generation (le nettoyage par sections
                        existe deja dans llama_cerveau.generer_riche).
4. Debugger récursif  : le code rate en sandbox est corrige en boucle
                        (max 3 essais) avec l'erreur exacte en retour —
                        les fautes de frappe ne sortent jamais.
5. Masqueur de        : le prompt systeme change selon la categorie
   personnalite         routee (calcul / faits / general) — la bonne
                        posture au bon moment, sans gros prompt fige.

Surete de conception :
- chaque agent est optionnel : echec = None ou texte inchange, le
  pipeline normal prend le relais (jamais bloquant) ;
- AURA_AGENTS=0 desactive tout (benchs, CI, debug) ;
- le planificateur ne se declenche que sur signature lexicale stricte.
"""
import logging
import os
import re

LOG = logging.getLogger("aura.agents")

_TICS = [
    "il est important de noter que ",
    "il convient de noter que ",
    "il faut noter que ",
    "il faut souligner que ",
    "il est a noter que ",
    "comme mentionne precedemment, ",
    "en d'autres termes, ",
    "en conclusion, on peut dire que ",
]
_TICS_RE = re.compile("|".join(re.escape(t) for t in _TICS), re.IGNORECASE)
_MAJ = re.compile(r"(^[a-zà-ÿ]|[.!?]\s+[a-zà-ÿ])")

_STOPWORDS = {
    "les", "des", "une", "est", "pour", "dans", "que", "qui", "quoi",
    "comment", "pourquoi", "avec", "sur", "aux", "cette", "sont", "the",
    "and", "what", "how", "quel", "quelle", "est-ce", "ce", "et", "de",
    "la", "le", "du", "un", "moi", "toi", "lui",
}

_TACHE_COMPLEXE = re.compile(
    r"\b(organis(e|ez|er|es)|pr[ée]par(e|ez|er|es)|planifi(e|ez|er|es)|"
    r"construis|construire|aide[- ]moi (a|à)|fais[- ]moi (un|une)|"
    r"redige[- ]moi|liste (d'|les |des |la )?etapes|etapes pour|"
    r"comment puis-je|projet de)\b", re.IGNORECASE)

_SYSTEME_PLAN = (
    "Tu es un planificateur. Ne reponds PAS a la demande : liste seulement "
    "2 a 3 etapes pour la realiser, une ligne par etape, au format exact "
    "'1. ...' '2. ...' '3. ...', sans aucun autre texte.")

_LIGNE_ENUM = re.compile(r"^\s*(\d)[\.\)]\s+(.+)$", re.MULTILINE)
# le 1B ecrit souvent « **Étape 1 : Préparation** » (markdown gras)
_LIGNE_ETAPE = re.compile(
    r"^\s*\**\s*[eé]tape\s*(\d)\s*[:.\-]\s*\**\s*(.+?)\**\s*$",
    re.MULTILINE | re.IGNORECASE)
_MD = re.compile(r"\*+|^-\s*|")                     # gras/puces residuels

# prompt systeme MINIMAL par categorie (le masque de personnalite) :
# court = RAM et budget de contexte presets pour le contenu, pas la posture
_PERSONNALITES = {
    "web": ("Tu es un journaliste factuel. Reponds UNIQUEMENT a partir des "
            "faits fournis (WEB / BASE DE FAITS) ; si l'information manque, "
            "dis-le franchement. Cite chiffres et dates."),
    "math": ("Tu es un expert en calcul et raisonnement numerique. "
             "Reponds droit au but, montre le calcul en une ligne."),
    "code": ("Tu es un expert Python. Donne du code propre et commente, "
             "avec un exemple d'utilisation en commentaire."),
    "general": ("Tu es un assistant clair et precis. Structure ta reponse, "
                "va a l'essentiel, n'invente aucun fait precis."),
}


def actifs() -> bool:
    """Interrupteur global : AURA_AGENTS=0 coupe les 5 agents d'un coup."""
    return os.environ.get("AURA_AGENTS", "1") != "0"


# ── Agent 1 : le Planificateur ──────────────────────────────────────────

def est_tache_complexe(question: str) -> bool:
    """Signature lexicale stricte : faux positif = juste un chemin normal."""
    return bool(_TACHE_COMPLEXE.search(question))


def planifier(question: str, max_tokens: int = 120) -> str | None:
    """Demande au cerveau de SEGMENTER avant de rediger (2-3 etapes).

    Renvoie le plan numerote, ou None si la demande n'est pas une tache
    ou si le modele ne produit pas un plan exploitable.
    """
    if not actifs() or not est_tache_complexe(question):
        return None
    from . import llama_cerveau                     # import paresseux
    brut = llama_cerveau.generer(question, systeme=_SYSTEME_PLAN,
                                 max_tokens=max_tokens)
    # deux formats tolérés : « 1. ... » et « **Étape 1 : ...** »
    vus: dict = {}
    for num, texte in (_LIGNE_ENUM.findall(brut or "")
                       + _LIGNE_ETAPE.findall(brut or "")):
        propre = _MD.sub("", texte).strip()
        if propre and num not in vus:
            vus[num] = propre
    etapes = [vus[k] for k in sorted(vus)][:3]
    if not 2 <= len(etapes) <= 3:
        LOG.info("[planificateur] pas de plan exploitable (%d etapes)",
                 len(etapes))
        return None
    plan = "\n".join(f"{i}. {e}" for i, e in enumerate(etapes, 1))
    LOG.info("[planificateur] tache decoupee en %d etapes", len(etapes))
    return plan


# ── Agent 2 : le Compresseur de contexte (filtre RAG) ───────────────────

def _tokens(texte: str) -> set:
    return {m for m in re.findall(r"[a-z0-9]{3,}", texte.lower())
            if m not in _STOPWORDS}


def compresser(texte: str, question: str, riche: bool = False) -> str:
    """Ne garder que le signal : 3 phrases utiles (mode simple), ou une
    coupe propre sur fin de phrase (mode riche). Texte court = intact."""
    if not actifs() or not texte:
        return texte
    if riche:
        if len(texte) <= 1500:
            return texte
        coupe = texte[:1500]
        point = max(coupe.rfind(". "), coupe.rfind("! "), coupe.rfind("? "))
        return coupe[:point + 1] if point > 400 else coupe

    if len(texte) <= 320:
        return texte
    phrases = [p.strip() for p in re.split(r"(?<=[.!?])\s+", texte) if p.strip()]
    utiles = [p for p in phrases if len(p) >= 25]
    if len(utiles) <= 3:
        return texte
    q_tokens = _tokens(question)
    scorees = []
    for p in utiles:
        score = len(_tokens(p) & q_tokens)
        if any(c.isdigit() for c in p) and any(c.isdigit() for c in question):
            score += 1
        scorees.append((score, p))
    garde = {p for _, p in sorted(scorees, key=lambda x: -x[0])[:3]}
    gardees = [p for p in phrases if p in garde]     # ordre original
    LOG.info("[compresseur] contexte web : %d -> %d phrases",
             len(phrases), len(gardees))
    return " ".join(gardees)


# ── Agent 3 : le Rédacteur (retire les tics du 1B) ──────────────────────

def nettoyer_style(texte: str) -> str:
    """Retire les formules toutes faites et les phrases dupliquees.

    « permet de » est volontairement CONSERVE : l'effacer au milieu d'une
    phrase casserait la grammaire (« cela permet de calculer »).
    """
    if not actifs() or not texte:
        return texte
    propre = _TICS_RE.sub("", texte)
    if propre != texte:
        # majuscule apres une formule retiree (debut de phrase / apres point)
        # — uniquement si un tic a ete retire : sinon on ne touche pas au texte
        propre = _MAJ.sub(lambda m: m.group(0).upper(), propre)
    # phrases dupliquees : on garde la premiere occurrence
    vues: set = set()
    uniques: list = []
    for phrase in re.split(r"(?<=[.!?])\s+", propre):
        cle = phrase.strip().lower()
        if cle and cle not in vues:
            vues.add(cle)
            uniques.append(phrase)
    nettoye = " ".join(uniques).strip()
    if nettoye != texte:
        LOG.info("[redacteur] style nettoye (%d -> %d car)",
                 len(texte), len(nettoye))
    return nettoye


# ── Agent 4 : le Debugger récursif (boucle de code) ─────────────────────

def boucle_code(question: str, max_tentatives: int = 3) -> str | None:
    """Genere -> sandbox -> si echec, re-genere avec l'erreur exacte.

    Le code qui rate ne sort JAMAIS : apres max_tentatives, None (le
    pipeline retombe sur la generation normale). Chaque relance montre
    au modele SON code + le message exact de la sandbox (self-debug).
    """
    from . import llama_cerveau, potcode            # imports paresseux
    erreur = ""
    dernier_code = ""
    for essai in range(1, max_tentatives + 1):
        if essai == 1:
            prompt = question
        else:
            prompt = (question
                      + "\n\nTa proposition precedente etait :\n"
                      + dernier_code
                      + "\n\nElle a ECHOUE en sandbox avec l'erreur : "
                      + erreur
                      + "\nCorrige ce defaut precis et redonne la fonction "
                        "resoudre + asserts, au format ```python.")
        try:
            brut = llama_cerveau.generer(prompt,
                                         systeme=potcode._SYSTEME_CODE,
                                         max_tokens=400)
        except Exception as e:
            LOG.info("[debugger] generation impossible (%s)", e)
            return None
        code = potcode.extraire_code(brut)
        if not code:
            erreur = "aucun bloc ```python avec une fonction resoudre"
            dernier_code = (brut or "")[:400]
            LOG.info("[debugger] essai %d/%d : pas de bloc python",
                     essai, max_tentatives)
            continue
        dernier_code = code
        try:
            rapport = potcode.verifier_code(code)
        except ValueError as e:
            erreur = str(e)
            LOG.info("[debugger] essai %d/%d refuse : %s",
                     essai, max_tentatives, e)
            continue
        correction = (f", {essai - 1} correction(s) auto." if essai > 1
                      else ".")
        LOG.info("[debugger] code valide a l'essai %d/%d", essai,
                 max_tentatives)
        return ("```python\n" + code + "\n```\n\n(Code verifie : "
                f"{rapport['nb_asserts']} asserts passes en sandbox"
                + correction + ")")
    LOG.info("[debugger] echec apres %d essais -> chemin normal",
             max_tentatives)
    return None


# ── Agent 5 : le Masqueur de personnalité ───────────────────────────────

def personnalite_pour(experts: set | None) -> str | None:
    """Masque de categorie (web > math > general). Usage interne."""
    if not actifs() or not experts:
        return None
    if "web" in experts:
        return _PERSONNALITES["web"]
    if "math" in experts:
        return _PERSONNALITES["math"]
    return _PERSONNALITES["general"]


def composer_personnalite(experts: set | None, question: str = "") -> str | None:
    """Prompt systeme complet = base Aura + masque de categorie.

    Jamais le masque seul : la base (_SYSTEME du cerveau) porte l'ancrage
    faits/formule. Les questions de logique renvoient None — elles gardent
    le suffixe CoT natif de generer() (ne pas retirer le raisonnement pas-
    a-pas aux questions ou le 1B en a le plus besoin).
    """
    masque = personnalite_pour(experts)
    if not masque:
        return None
    from .llama_cerveau import _MOTS_LOGIQUE, _SYSTEME as BASE
    if any(m in question.lower() for m in _MOTS_LOGIQUE):
        return None
    return BASE + " " + masque
