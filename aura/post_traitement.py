"""Post-traitement du texte GENERE par le cerveau : fonctions PURES.

Deux garde-fous vivaient dans `aura/llama_cerveau.py`, au milieu du CHARGEMENT
du modele alors qu'ils ne partagent rien avec lui :

1. DISJONCTEUR ANTI-REPETITION : le 1B recopie une phrase ou tourne en boucle
   sur un n-gramme — il faut le voir (`_en_boucle`) et couper (`_couper_boucle`).
2. CANALISATION : le 1B RECLAME son contexte au lieu de repondre, ou recite le
   prompt — `_sans_parrotage` retire ces fuites avant livraison.

Ici, tout est PUR (`texte -> texte` ou `texte -> bool`) : aucun etat de module,
aucune E/S, aucun appel au modele. `aura/llama_cerveau.py` les REEXPORTE : son
espace de noms reste l'API historique (`llama_cerveau._en_boucle(...)`), y
compris pour un `monkeypatch.setattr` dans un test.
"""
import re


# ── DISJONCTEUR ANTI-REPETITION ───────────────────────────────────
# Bug vécu : la partie 2 a recopié la partie 1 mot pour mot, la phrase
# « La notion de puissance est souvent… » est revenue 3 fois, et la
# generation a tourné pendant des dizaines de secondes dans le vide.
# Ici : detection (phrase repetee / n-gramme en boucle), COUPURE IMMEDIATE
# du stream, puis UNE relance avec penalite de repetition + temperature
# differente. Jamais la boucle n'est livree telle quelle.

_SEUIL_PHRASE = 50        # phrase « longue » candidate a la copie
_SEUIL_NB_PHRASES = 2     # occurrences de cette phrase = boucle
_SEUIL_NB_NGRAM = 5       # sequences de 4 mots revues 5 fois = boucle
_SEUIL_FRAGMENT = 40      # fenetre de 40 signes : toute suite partagee
                          # d'au moins 45 signes = copie (cf. _deja_dit)
_PAS_FRAGMENT = 5         # pas de glissement de la fenetre (5 garantit
                          # la detection des copies de 45 signes+)


def _phrases(texte: str) -> list[str]:
    """Decoupe en phrases / lignes (les balises du plan comptent comme lignes)."""
    return [p.strip() for p in re.split(r"(?<=[.!?…])\s+|\n+", texte or "")
            if p.strip()]


def _deja_dit(fragment: str, bas: str) -> bool:
    """Vrai si `fragment` (minuscules) est deja present dans `bas`.

    Mot pour mot (copie verbatim) OU par FENETRE de 40 signes glissee de
    5 en 5 : une phrase reprise avec un autre premier mot (« Pourtant,
    cette question souleve un debat... » puis « En definitive, cette
    question souleve un debat... ») est une copie, meme si la comparaison
    exacte echoue. Toute suite partagee d'au moins 45 signes est vue.
    """
    if not fragment or not bas:
        return False
    if fragment in bas:
        return True
    n = len(fragment)
    if n < _SEUIL_FRAGMENT:
        return False
    for i in range(0, n - _SEUIL_FRAGMENT + 1, _PAS_FRAGMENT):
        if fragment[i:i + _SEUIL_FRAGMENT] in bas:
            return True
    return False


_RE_FIN_PHRASE = re.compile(r"[.!?…][\"»')\]]*\s*$")
# Mots d'accroche qui ne terminent JAMAIS une phrase : leur presence en
# fin de texte signale une troncature sur max_tokens (cas observe au
# banc : « ... la puissance de la technologie est »).
_RE_QUEUE_MORTE = re.compile(
    r"(?:['’]|[-–—]|\b[a-z]|\b(?:et|ou|mais|donc|car|de|des|du|la|le|les|un|une|"
    r"en|dans|pour|avec|sur|par|chez|vers|jusque|qui|que|quoi|dont|"
    r"ce|cette|est|sont|etre|se)\s*)$", re.I)


def _finir_phrase(texte: str, forcer: bool = False) -> str:
    """Evite de livrer une phrase INCOMPLETE en fin de texte.

    - forcer=True : coupe apres le dernier . ! ? … (le disjoncteur coupe
      au milieu d'une phrase, parfois au milieu d'un mot : « ... et d' ») ;
    - sinon : ne coupe que si la fin sent la troncature (mot d'accroche,
      apostrophe, tiret) — « ... gagne du terrain » reste intact.

    Un texte deja ponctue, ou sans ponctuation aucune, est rendu tel quel :
    jamais de texte complet jete par erreur.
    """
    t = (texte or "").strip()
    if not t or _RE_FIN_PHRASE.search(t):
        return t
    if not forcer and not _RE_QUEUE_MORTE.search(t):
        return t
    fins = list(re.finditer(r"[.!?…]", t))
    if not fins:
        return t
    coupe = t[:fins[-1].end()].rstrip()
    return coupe if coupe else t


def _en_boucle(texte: str) -> bool:
    """Vrai si le texte CONTIENT une boucle de generation.

    Deux signaux, volontairement conservateurs (faux positif = une relance
    inutile, jamais une regression) :
      1. une phrase de >= 50 signes apparaît au moins 2 fois
         (le cas observe : copie inter-sections),
      2. une sequence de 4 mots revient >= 5 fois (boucle serree).
    """
    if not texte:
        return False
    vus: dict[str, int] = {}
    for p in _phrases(texte):
        if len(p) >= _SEUIL_PHRASE:
            cle = p.lower()
            vus[cle] = vus.get(cle, 0) + 1
            if vus[cle] >= _SEUIL_NB_PHRASES:
                return True
    mots = re.findall(r"[\w'’]+", (texte or "").lower())
    if len(mots) >= 40:
        ng: dict[tuple, int] = {}
        for i in range(len(mots) - 3):
            cle_ng = tuple(mots[i:i + 4])
            ng[cle_ng] = ng.get(cle_ng, 0) + 1
            if ng[cle_ng] >= _SEUIL_NB_NGRAM:
                return True
    return False


def _couper_boucle(texte: str) -> str:
    """Tronque AVANT la deuxieme occurrence de la phrase repetee.

    Plan B du disjoncteur : si la relance derape aussi, on livre la version
    courte (une phrase perdue vaut mieux qu'une boucle affichee).
    """
    vus: set[str] = set()
    for m in re.finditer(r"[^.!?\n]{%d,}[.!?]" % _SEUIL_PHRASE, texte or ""):
        cle = m.group(0).strip().lower()
        if cle in vus:
            # la coupure tombe AU MILIEU d'une phrase : on la termine
            # proprement (jamais « ... et d' » livree a l'utilisateur)
            coupe = (texte[:m.start()]).strip()
            return _finir_phrase(coupe, forcer=True) or coupe
        vus.add(cle)
    return (texte or "").strip()


def _sans_doublons(texte: str) -> str:
    """Retire les phrases repetees AU SEIN d'un meme texte.

    Mot pour mot OU par fragment (45+ signes partages avec un debut de
    phrase different) : utile pour les extraits web (deux resultats
    colles cote a cote) et en garde-fou final avant livraison. La mise en
    page est preservee : si rien n'est a retirer, le texte original sort
    tel quel (sinon les paragraphes \n\n le restent).
    """
    if not texte:
        return texte
    propres: list[str] = []
    bas = ""
    touche = False
    for bloc in re.split(r"\n{2,}", texte):
        phrases = _phrases(bloc)
        gardes: list[str] = []
        for p in phrases:
            if len(p) >= _SEUIL_PHRASE and _deja_dit(p.lower(), bas):
                touche = True            # copie (verbatim ou fragment)
                continue
            gardes.append(p)
            bas = (bas + "\n" + p.lower()) if bas else p.lower()
        if not gardes:
            if bloc.strip():
                touche = True            # bloc entier copie -> jete
            continue
        propres.append(bloc if len(gardes) == len(phrases)
                       else " ".join(gardes))
    if not touche:
        return texte                     # rien a retirer : texte intact
    propre = "\n\n".join(propres).strip()
    return propre or texte


_RE_LIBELLE_PLAN = re.compile(
    r"\*{0,2}(?:SECTION\s+A\s+REDIGER\s*:|PARTIE\s*[1-9]\s*[:\-–]|"
    r"CONNECTEURS\s*:)", re.I)


def _sans_libelles(texte: str) -> str:
    """Retire les libelles de plan recopies dans la redaction finale.

    Les CONSIGNES D'OUTIL du prompt systeme (« SECTION A REDIGER :
    PARTIE n », « CONNECTEURS : ») sont parfois recopiees par le 1B en tete
    de sa reponse (observe au banc : un item KO sur cette balise ecolee).
    La ligne partant du libelle est jetee ; sauf si elle est fermee par **
    et suivie de vrai texte, auquel cas on garde ce qui suit la fermeture.
    """
    if not texte:
        return texte
    lignes = []
    for ligne in texte.split("\n"):
        while True:
            m = _RE_LIBELLE_PLAN.search(ligne)
            if not m:
                break
            suite = ligne[m.end():]
            cloture = re.search(r"\*\*", suite)
            if cloture:
                # entete en gras fermee puis texte reel -> on garde la suite
                ligne = ligne[:m.start()] + suite[cloture.end():]
                continue
            # pas de fermeture : tout ce qui suit est le libelle -> la
            # fin de la ligne saute (le vrai texte est sur les suivantes)
            ligne = ligne[:m.start()]
            break
        lignes.append(ligne)
    return "\n".join(lignes).strip()

# ── CANALISATION : quand le 1B RECLAME son contexte au lieu de repondre ──
# Mesure du 12 oct. 2026, question « Qui a gagne le tournoi de tennis de
# Roland-Garros en 2025 ? » avec un contexte web de 3535 car. POURTANT
# present (3 extraits verifies) :
#   « J'ai besoin des informations pour repondre a ta question. Dis-moi :
#     "Qu'est-ce qui est fourni ci-dessous et est pertinent" (WEB / BASE DE
#     FAITS) ... dis-moi ce que c'est. »
# 82 s de calcul pour rendre la question a l'utilisateur. Des que le bloc
# joint ne ressemble pas a ce que le 1B attend, il recite son prompt
# systeme. Deux degats, deux remedes distincts :
#   1. les ETIQUETTES du prompt (WEB / BASE DE FAITS, FAITS WEB, DONNEES
#      WEB, FORMULE EXACTE, SOURCES WEB, SECTION A REDIGER) fuient en clair,
#      souvent en majuscules : elles partent MEME au milieu d'une phrase qui,
#      elle, reste ;
#   2. la PHRASE ENTIERE qui demande les informations au lecteur ne repond a
#      rien : elle saute.
# Si tout le texte etait du bavardage, on rend "" — c'est volontaire :
# l'appelant sait qu'il doit servir autre chose (les extraits verifies)
# plutot qu'une non-reponse.
_RE_ETIQUETTE_PROMPT = re.compile(
    r"\s*\(?\b(?:WEB\s*/\s*BASE DE FAITS|FAITS WEB|BASE DE FAITS|"
    r"DONNEES WEB|FORMULE EXACTE|SOURCES WEB|SECTION A REDIGER)\b\)?\s*",
    re.IGNORECASE)
_RE_RECITATION_PROMPT = re.compile(
    r"ci-dessous et (?:est|sont) pertinent|appuie[- ]toi dessus|"
    r"j'utiliserai ce bloc comme contexte", re.IGNORECASE)
#: tournures de DEMANDE : elles n'ont de sens que dans un prompt
#: systeme (« Dis-moi X »), jamais dans une reponse rendue a l'utilisateur.
_RE_DEMANDE_CONTEXTE = re.compile(
    r"\bdis[- ]moi\b|\bdites[- ]moi\b|\bindique[- ]moi\b|"
    r"\bfournis[- ]moi\b|\bdonne[- ]moi\b|"
    r"\bpeux[- ]tu me (?:fournir|donner|preciser|dire|indiquer)\b|"
    r"\bpouvez[- ]vous me (?:fournir|donner|preciser|dire|indiquer)\b|"
    # « J'ai besoin des informations / des sources / du titre ... pour
    # repondre » : la 2e mesure (12 oct. 2026) a produit la variante
    # « J'ai besoin des SOURCES pour repondre » — le nom du besoin change,
    # la tournure reste, donc on la reconnait sur la tournure.
    r"j'ai besoin\s+[^.\n]{0,40}?pour\s+(?:te\s+|vous\s+)?r[eé]pondre|"
    r"\bquelle est (?:ta|la|votre) question\b", re.IGNORECASE)
#: constat d'absence : la phrase ne saute QUE s'il porte sur des donnees
#: (« je ne vois pas d'autre solution » est une reponse, pas du bavardage).
_RE_ABSENCE_DONNEE = re.compile(
    r"je ne vois pas|je n'ai pas|il n'y a pas|\baucun", re.IGNORECASE)
_RE_MOT_DONNEE = re.compile(
    r"\b(?:contexte|information|donn[ée]e|source|extrait|document|preuve|"
    r"r[ée]sultat)s?\b", re.IGNORECASE)
#: LA SUITE D'UNE PHRASE RECITEE n'est souvent qu'un reliquat de citation
#: (« ... le recap des champions ! | Source : https://... ») : quand une
#: demande d'information vient d'etre retiree, ce qui la suit immediatement
#: et ressemble a une etiquette de source ou a une URL part avec elle.
_RE_RELIQUAT = re.compile(r"^\s*[|;•·:\-–—]|^\s*Source\s*:|https?://",
                         re.IGNORECASE)


def _parrotage(phrase: str) -> bool:
    """Vrai si la phrase RECLAME le contexte au lieu de repondre."""
    if _RE_DEMANDE_CONTEXTE.search(phrase):
        return True
    if _RE_RECITATION_PROMPT.search(phrase):
        return True
    return bool(_RE_ABSENCE_DONNEE.search(phrase)
                and _RE_MOT_DONNEE.search(phrase))


def _sans_parrotage(texte: str) -> str:
    """Retire les recitations de prompt et les demandes d'information.

    Rend le texte INCHANGE (au caractere pres) quand il ne contient rien de
    tel : ce filtre ne doit jamais abimer une reponse normale. Rend ""
    quand tout le texte etait du bavardage — l'appelant sert alors les
    extraits verifies (cf. `orchestrateur._repli_extractif`).
    """
    if not texte:
        return texte
    if not (_RE_ETIQUETTE_PROMPT.search(texte)
            or any(_parrotage(p) for p in _phrases(texte))):
        return texte
    propre = re.sub(r"[ \t]{2,}", " ", _RE_ETIQUETTE_PROMPT.sub(" ", texte))
    lignes = []
    for ligne in propre.split("\n"):
        gardees = []
        fuite_vue = False
        for piece in _phrases(ligne):
            if _parrotage(piece):
                fuite_vue = True                  # sa suite peut etre un reliquat
                continue
            if fuite_vue and _RE_RELIQUAT.search(piece):
                continue
            gardees.append(piece)
        lignes.append(" ".join(gardees).strip())
    return "\n".join(lignes).strip()
