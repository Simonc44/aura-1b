"""Les 5 agents cognitifs d'RontoGPT : petite tete, grands gestes.

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
                        apres generation (nettoyage de style, deduplication
                        et coupure des boucles, cf. llama_cerveau).
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
import unicodedata

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
    "code": ("Tu es un assistant de programmation. Fournis le code demandé "
             "dans le langage précisé ; si aucun langage n'est précisé, utilise "
             "Python. Donne un exemple d'utilisation bref. N'affirme pas que le "
             "code a été exécuté ou testé si ce n'est pas le cas."),
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
    normalise = unicodedata.normalize("NFKD", (texte or "").lower())
    normalise = "".join(c for c in normalise
                        if not unicodedata.combining(c))
    return {m for m in re.findall(r"[a-z0-9]{3,}", normalise)
            if m not in _STOPWORDS}


def compresser(texte: str, question: str) -> str:
    """Ne garder que les extraits avec un signal lexical de pertinence.

    Un contexte de recherche sans aucun recouvrement lexical est écarté :
    les petits modèles peuvent être distraits par un RAG hors sujet. Le gate
    est volontairement peu coûteux et n'affirme pas que le texte retenu
    suffit à répondre.
    """
    if not actifs() or not texte:
        return texte
    structure_web = bool(re.search(r"\b(?:titre|extrait)\s*:", texte,
                                   re.IGNORECASE))
    if len(texte) <= 320 and not structure_web:
        return texte
    q_tokens = _tokens(question)
    if not q_tokens:
        return texte
    # Les résultats DDG sont séparés par ligne. Pour les autres contextes,
    # on retombe sur les phrases afin de garder le compresseur générique.
    lignes = [p.strip() for p in texte.splitlines() if p.strip()]
    morceaux = (lignes if len(lignes) > 1 else
                [p.strip() for p in re.split(r"(?<=[.!?])\s+", texte)
                 if p.strip()])
    scorees = []
    for p in morceaux:
        score = len(_tokens(p) & q_tokens)
        if any(c.isdigit() for c in p) and any(c.isdigit() for c in question):
            score += 1
        scorees.append((score, p))
    pertinents = [(score, p) for score, p in scorees if score > 0]
    if not pertinents:
        LOG.info("[compresseur] contexte ecarte : aucun terme pertinent")
        return ""
    # Une source courte ne mérite pas de troncature, mais on exclut quand
    # même les lignes sans signal lexical. Les contextes longs sont limités
    # aux trois unités les mieux notées pour préserver la fenêtre du modèle.
    # BRIDAGE (fix bug 1) : jamais plus de 3 morceaux (10 pour une
    # recherche complexe), meme sur un contexte court.
    try:
        from . import memoire_web
        plafond = memoire_web.limite_morceaux(question)
    except Exception:       # noqa: BLE001 - le compresseur ne doit pas casser
        plafond = 3
    limites = min(len(pertinents), plafond)
    garde = {p for _, p in sorted(pertinents,
                                  key=lambda x: -x[0])[:limites]}
    gardees = [p for p in morceaux if p in garde]
    LOG.info("[compresseur] contexte : %d -> %d extraits pertinents",
             len(morceaux), len(gardees))
    return "\n".join(gardees)


# ── Agent 2b : Triplets sémantiques (RAG → graphe, hyper-compression) ───

# verbes de relation courants : un extrait « X est la capitale de Y »
# devient le triplet (X, capitale_de, Y) — 10x plus court qu'une phrase
_RELATIONS = (
    ("est la capitale de", "capitale_de"),
    ("est le capital de", "capitale_de"),
    ("capitale de", "capitale_de"),
    ("est le president de", "president_de"),
    ("premier ministre de", "premier_ministre_de"),
    ("a ete fondee en", "fondee_en"),
    ("fonde en", "fonde_en"),
    ("fondee en", "fondee_en"),
    ("situ[eé]e? dans", "situe_dans"),
    ("situ[eé]e? en", "situe_en"),
    ("appartient a", "appartient_a"),
    ("est connu pour", "connu_pour"),
    ("mesure", "mesure"),
    ("pese", "pese"),
    ("habite", "habite"),
    ("est ne[eé]? en", "ne_en"),
    ("est mort en", "mort_en"),
    ("invente par", "invente_par"),
    ("decouvert par", "decouvert_par"),
    ("signifie", "signifie"),
)


def en_triplets(texte: str, question: str, max_triplets: int = 6) -> str:
    """Traduit le contexte web en triplets (sujet | relation | objet).

    La memoire de travail du 1B est courte : des triplets de 6-10 tokens
    remplacent des phrases de 30 — le graphe de faits gagne au passage
    des faits verifies web (compatibles graphe_faits.py, format s/r/o).
    Texte deja court ou sans relation detectee -> renvoye intact.
    """
    if not actifs() or not texte or len(texte) < 80:
        return texte
    triplets: list = []
    for morceau in re.split(r"(?<=[.!?])\s+|\n", texte):
        morceau = morceau.strip(" -|")
        if len(morceau) < 15:
            continue
        basse = morceau.lower()
        for motif, relation in _RELATIONS:
            m = re.search(motif, basse)
            if not m:
                continue
            sujet = morceau[:m.start()].strip(" ,.;:|")
            objet = morceau[m.end():].strip(" ,.;:|")
            if len(sujet) < 2 or len(objet) < 2 or len(objet) > 80:
                continue
            triplets.append(f"{sujet} | {relation} | {objet}")
            break
        if len(triplets) >= max_triplets:
            break
    if not triplets:
        return texte
    LOG.info("[triplets] contexte : %d -> %d car (%d triplets)",
             len(texte), sum(len(t) for t in triplets), len(triplets))
    return "FAITS (format sujets | relation | objet) :\n" + "\n".join(triplets)


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
    """Prompt systeme complet = base RontoGPT + masque de categorie.

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


# ── Agent 6 : le Mimétisme (few-shot, style copié) ──────────────────────

# Briques d'exemples « calibre grand modèle » : le 1B n'invente pas le
# style, il le COPIE — quelques démonstrations sous les yeux déplacent
# son attention vers les tournures qu'on attend (effet few-shot).
#
# JAMAIS de marqueurs de tournure « Q: » / « R: » : le 1B les prenait
# pour le format à CONTINUER et fabriquait un tour d'interlocuteur. Sur
# une entrée faible (« ok ») il rendait « R: Quel est le taux de change
# du euro à la date limite précédente ? », et en prod il recopiait
# l'exemple du ciel en l'attribuant à l'utilisateur (« Vous avez posé :
# Pourquoi le ciel est-il bleu ? ») — bug réel reproduit, 2026-10.
# Les exemples sont donc des CITATIONS de style, explicitement
# interdites de recopie. (Mesuré : l'ancien correctif — déplacer la
# brique dans le prompt système — n'avait PAS suffi.)
_EXEMPLES_STYLE = {
    "general": (
        "Exemples de style attendu — ce sont de simples échantillons de "
        "tournure, PAS des questions à traiter et PAS des réponses à "
        "recopier :\n"
        "« La lumière solaire rencontre l'atmosphère et ses molécules "
        "dispersent davantage les longueurs d'onde courtes : le bleu "
        "domine notre ciel. Ce phénomène, la diffusion de Rayleigh, "
        "explique aussi le rouge des crépuscules. »\n"
        "« L'épargne constitue un rempart contre l'imprévu et le point "
        "de départ de tout projet. Placée intelligemment, elle travaille "
        "pour vous : le temps transforme la régularité en patrimoine. »\n\n"
        "Rédige TA réponse à la question posée en gardant ce ton : "
        "précis, fluide, sans formule toute faite."
    ),
    "web": (
        "Exemples de style attendu — ce sont de simples échantillons de "
        "tournure, PAS des questions à traiter et PAS des réponses à "
        "recopier :\n"
        "« La NASA a annoncé le retard de sa mission lunaire Artemis : "
        "le calendrier glisse de plusieurs mois. En parallèle, SpaceX "
        "enchaîne les tirs de son Starship géant. »\n"
        "« L'inflation ralentit en zone euro, selon Eurostat, mais "
        "reste au-dessus de la cible. Les marchés anticipent désormais "
        "une baisse des taux. »\n\n"
        "Rédige TA réponse sur ce modèle : faits d'abord, chiffres si "
        "disponibles, aucune digression."
    ),
    "math": (
        "Exemples de style attendu — ce sont de simples échantillons de "
        "tournure, PAS des questions à traiter et PAS des réponses à "
        "recopier :\n"
        "« 391. Décomposition : 17 × 20 = 340, 17 × 3 = 51, total 391. »\n"
        "« 30. »\n\n"
        "Rédige TA réponse sur ce modèle : résultat d'abord, étape "
        "courte ensuite, rien d'autre."
    ),
}

# analyse rapide de la question pour choisir la brique la plus utile
_MOTS_WEB = ("actualit", "aujourd'hui", "dernier", "2024", "2025", "2026",
             "prix", "cours", "news")
_MOTS_MATH = ("combien", "calcul", "somme", "produit", "pourcent", "%",
              "racine", "divis")


def exemple_style(question: str, categorie: str = "") -> str | None:
    """Retourne la brique few-shot adaptee (mimétisme de style)."""
    if not actifs():
        return None
    cle = categorie or (
        "math" if any(m in question.lower() for m in _MOTS_MATH)
        else "web" if any(m in question.lower() for m in _MOTS_WEB)
        else "general")
    return _EXEMPLES_STYLE.get(cle)
