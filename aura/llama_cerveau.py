"""Cerveau Llama 3.2 1B Instruct — optimise CPU (flash attention, KV cache q8_0).

Optimisations CPU reelles (benchmarkees sur ce PC) :
- flash_attn=True          : attention fusionnee, moins de passes memoire
- type_k=8, type_v=8       : KV cache q8_0 -> RAM cache divisee par 2,
                             conversations plus longues a RAM egale
- n_ctx=1024, n_batch=512  : prefill x2, parallelisme des 6 coeurs
- n_threads=os.cpu_count() : +26% vs coeurs physiques seuls

Optimisations d'intelligence :
- quantification Q4_K_M (defaut, mesuree 3/3 en raisonnement) ou Q6_K
  (option, fidele aux poids) via variable AURA_GGUF
- raisonnement etape par etape injecte pour les questions de logique
- max_tokens adaptatif : court pour les faits, long pour les explications
- memoire de conversation (historique multi-tours) via parametre historique
"""
import logging
import os
import re
import time

LOG = logging.getLogger("aura.llama")

_DOSSIER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "modeles")

# Q4_K_M par defaut : mesure 3/3 en raisonnement + 10% plus rapide que Q6_K.
# AURA_GGUF=Q6_K pour la fidelite maximale aux poids originaux.
# CERVEAUX — bench officiel (scripts/comparer_cerveaux.py, meme quiz) :
#   Llama 3.2 1B Q4_K_M : 6/7 a ~3.0 s/reponse  ← DEFAUT
#   MiniCPM5-1B Q4_K_M  : 3/7 a ~10.2 s (cerveau « thinking-only » : sans
#   sa reflexion native il decroche, avec elle il est 3x plus lent)
# AURA_GGUF=MINICPM (ou MINICPM_Q8) pour basculer, AURA_THINK=1 pour la
# reflexion native (puzzles difficiles).
_CHEMINS = {
    "Q4_K_M": os.path.join(_DOSSIER, "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "Q6_K": os.path.join(_DOSSIER, "Llama-3.2-1B-Instruct-Q6_K.gguf"),
    "MINICPM": os.path.join(_DOSSIER, "MiniCPM5-1B-Q4_K_M.gguf"),
    "MINICPM_Q8": os.path.join(_DOSSIER, "MiniCPM5-1B-Q8_0.gguf"),
}
_CHEMIN_GGUF = _CHEMINS[os.environ.get("AURA_GGUF", "Q4_K_M").upper()]

# Candidats de repli : le GGUF peut vivre dans le cache du kernel (.aef)
# plutot que dans modeles/ (machine qui n'a que le .aef, ou doublon evite).
# AURA_GGUF_CHEMIN impose un emplacement precis.
#  - source tree      : projet/aura/../.cache_aef/cerveau.gguf
#  - code extrait     : .cache_aef/code/aura/../../cerveau.gguf  (kernel)
# Chaque candidat ne gagne que s'il existe reellement sur le disque.
_dossier_aura = os.path.dirname(os.path.abspath(__file__))
_CANDIDATS = [
    os.environ.get("AURA_GGUF_CHEMIN"),
    _CHEMIN_GGUF,
    os.path.join(_dossier_aura, "..", ".cache_aef", "cerveau.gguf"),
    os.path.join(_dossier_aura, "..", "..", "cerveau.gguf"),
]
# le cache du kernel peut contenir l'ancien cerveau : si le cerveau demande
# est absent mais qu'un GGUF du cache existe, c'est lui qui sert (le .aef
# publie embarquait Llama) — sinon, machine neuve sans le bon fichier,
# _charger() levera l'erreur explicite.
_CHEMIN_GGUF = next((os.path.abspath(c) for c in _CANDIDATS
                     if c and os.path.exists(c)), _CHEMIN_GGUF)

# Config compilee du kernel (.aef) : prioritaire sur les reglages par defaut.
# Elle contient les reglages AUTO-TUNES pour la machine de forge (profil
# materiel detecte a la forge). AURA_TUNING=1 force l'auto-tuning LOCAL
# (machine differente de celle de forge, ex : quelqu'un qui execute le .aef).
if os.environ.get("AURA_CONFIG"):
    try:
        import json as _json
        _cfg = _json.loads(os.environ["AURA_CONFIG"])
        _CHEMIN_GGUF = _cfg.get("chemin_gguf", _CHEMIN_GGUF)
    except Exception:
        pass


def _reglages_materiel() -> dict:
    """Reglages llama.cpp : config compilee du .aef, ou auto-tuning local."""
    if os.environ.get("AURA_TUNING"):
        try:
            from . import profil_materiel as pm
            reg = pm.reglages_optimaux()
            LOG.info("[profil] auto-tuning local : %s", reg)
            return reg
        except Exception:
            pass
    try:
        import json as _json
        cfg = _json.loads(os.environ.get("AURA_CONFIG", "{}"))
        if cfg:
            return {"n_ctx": cfg.get("n_ctx", 1536),
                    "n_batch": cfg.get("n_batch", 768),
                    "n_threads": cfg.get("n_threads", os.cpu_count() or 4),
                    "n_threads_batch": cfg.get("n_threads_batch",
                                               cfg.get("n_threads", 4)),
                    "flash_attn": cfg.get("flash_attn", True),
                    "type_kv": cfg.get("type_kv", 8)}
    except Exception:
        pass
    # ni .aef ni tuning : profil de la machine locale
    try:
        from . import profil_materiel as pm
        return pm.reglages_optimaux()
    except Exception:
        n = os.cpu_count() or 4
        return {"n_ctx": 1536, "n_batch": 768, "n_threads": n,
                "n_threads_batch": n, "flash_attn": True, "type_kv": 8}

_SYSTEME = (
    "Tu es Aura, une IA hybride locale specialisee en maths exactes et faits temps reel. "
    "Reponds toujours a la question posee, en francais si la question est en francais, "
    "en anglais sinon. Sois concise et utile. "
    "Si des FAITS WEB ou une FORMULE sont fournis ci-dessous et sont pertinents, "
    "appuie-toi dessus. Sinon, reponds avec tes propres connaissances. "
    "Si on te demande qui tu es, presente-toi comme Aura."
)

# declencheurs de raisonnement pas-a-pas : les questions de logique/relations
# sont la faiblesse des petits modeles -> on force la decomposition.
_MOTS_LOGIQUE = ("plus que", "moins que", "si ", "alors", "avant", "apres",
                 "pourquoi", "deduis", "en deduis", "logique", "ordre",
                 "qui est le plus", "quel age", "différence entre", "difference entre")

# ── Étape 3 : prompts de la generation multi-pass ─────────────────────

# FEW-SHOT AGRESSIF de redaction abstraite (contexte appris en une fois,
# reutilise tel quel) : le 1B ne doit PAS inventer la structure d'une
# dissertation — il COPIE une demonstration rigoureuse deja ecrite
# (theses / antitheses / synthese), le few-shot deplace son attention
# vers ce gabarit eprouve.
_EXEMPLE_DISSERTATION = (
    "Exemple de redaction abstraite attendue (imite cette structure, "
    "adapte le contenu) :\n"
    "Q: La technique libere-t-elle l'homme ?\n"
    "R: La question divide : la technique apparait d'abord comme une "
    "affranchissement. En mechanisant les corvees, elle rend a l'homme "
    "un temps que la nature lui refusait ; le tracteur affranchit le "
    "paysan, la machine a laver la maison.\n"
    "Pourtant ce meme instrument asservit. Celui qui ne maitrise pas "
    "l'outil en devient le serviteur : cadences imposees, notifications "
    "permanentes, competences perimees en une decennie. La liberation "
    "d'hier forge la dependance de demain.\n"
    "Au fond, la technique n'affranchit que celui qui l'interroge. Elle "
    "amplifie la liberte deja conquise au lieu de la donner : elle est "
    "un pouvoir, non une grace.\n"
    "CONSIGNE : pose la these (PARTIE 1 : UNE affirmation + preuve), "
    "suis-la de son antithese (PARTIE 2 : UNE negation + exemple), "
    "termine par une synthese personnelle (PARTIE 3) qui depasse "
    "l'opposition. Chaque partie : 60 a 110 mots, une idee par phrase."
)

# Message SYSTEME de la passe plan : l'exemple y vit avec sa regle
# anti-copie. Place en systeme (et non colle a la question), le 1B
# l'utilise comme GABARIT de structure sans recopier son theme —
# contamination observee (« la technique libere l'homme » reecrit pour
# une question sur le bonheur) quand l'exemple etait dans le prompt user.
_SYSTEME_PLAN = (
    "Tu rediges des analyses en dissertation rigoureuse : these "
    "(affirmation prouvee), antithese (negation nuancee), synthese "
    "(depassement personnel).\n\n"
    + _EXEMPLE_DISSERTATION
    + "\nREGLE ABSOLUE : l'exemple illustre uniquement la STRUCTURE. "
      "Ne reprends JAMAIS son theme (la technique, la liberte de "
      "l'homme). Traite EXCLUSIVEMENT le theme de la question posee."
)

_PROMPT_PLAN = (
    "En te basant sur ces faits :\n{contexte}\n\n"
    "et sur cette question : {question}\n\n"
    "Genere UNIQUEMENT le plan de ta reponse, au format exact :\n"
    "PARTIE 1 : <LA these : l'affirmation, avec sa preuve>\n"
    "PARTIE 2 : <L'ANTITHESE : la negation nuancee, avec son exemple>\n"
    "PARTIE 3 : <LA SYNTHESE : ce qui depasse l'opposition>\n"
    "CONNECTEURS : <5 connecteurs logiques avances separes par des virgules>"
)

_PROMPT_SECTION = (
    "Redige la section {numero}/{total} de l'analyse de : {question}\n"
    "SECTION A REDIGER : {titre}\n"
    "{role_section}\n"
    "Deja ecrit (RESUME — coherence, PAS un modele a copier) :\n"
    "{memoire}\n"
    "Faits disponibles :\n{contexte_court}\n"
    "CONSIGNES : vocabulaire riche et precis, phrases courtes reliees par "
    "des connecteurs logiques, 60 a 110 mots, sans titres ni balises.\n"
    "INTERDICTION ABSOLUE de reprendre une phrase (meme debut de phrase) "
    "des parties deja ecrites : ecris un contenu entierement nouveau. "
    "Ne recopie AUCUN libelle du prompt (SECTION A REDIGER, PARTIE n...)."
)

# Role rhetorique de chaque section : la partie 1 AFFIRME et prouve, la
# 2 NIE avec nuance et exemple, la 3 DEPASSE l'opposition. Dire au 1B
# CE QUE fait chaque partie (et pas seulement son titre) est ce qui
# transforme un plan en vraie dissertation.
_ROLES_SECTION = {
    1: ("ROLE : AFFIRMER. Pose la these nettement des la premiere phrase, "
        "puis donne UNE preuve concrete qui l'appuie. Aucune nuance ici."),
    2: ("ROLE : CONTRER. Ouvre par un connecteur d'opposition (Pourtant, "
        "Cependant, Neanmoins). Expose la limite ou l'envers de la these, "
        "avec UN exemple NOUVEAU qui la rend tangible. N'emprunte aucun "
        "exemple a la section 1, ne reutilise aucune de ses tournures."),
    3: ("ROLE : DEPASSER. Ouvre par un connecteur de synthese (Au fond, "
        "En definitive, Tout se joue alors). Formule le depassement "
        "personnel : ce que la these et l'antithese revelent ensemble. "
        "Pas de conclusion molle du type 'il y a du pour et du contre'. "
        "Interdit de recopier des phrases entieres des sections "
        "precedentes : reformule avec d'autres mots."),
}

_PROMPT_STYLE = (
    "Tu es un redacteur litteraire et scientifique de haut niveau.\n"
    "Tu rediges en dissertation rigoureuse : these, antithese, synthese.\n"
    "En te basant sur ce plan :\n{plan}\n\n"
    "Redige l'analyse finale de la question : {question}\n\n"
    "CONSIGNES DE STYLE STRICTES :\n"
    "- PARTIE 1 affirme et prouve ; PARTIE 2 contredit avec nuance ; "
    "PARTIE 3 depasse l'opposition (jamais de 'il y a du pour et du "
    "contre').\n"
    "- Utilise un vocabulaire riche, precis et varie (evite les mots valises "
    "comme 'faire', 'dire', 'chose').\n"
    "- Fais des phrases courtes mais percutantes, reliees par les connecteurs "
    "logiques listes.\n"
    "- Interdiction de repeter la meme structure de phrase.\n"
    "- Integre naturellement les faits suivants : {contexte_court}\n"
    "- Reponds en francais, sans balises ni commentaires."
)

_llm = None

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


def _phrases(texte: str) -> list[str]:
    """Decoupe en phrases / lignes (les balises du plan comptent comme lignes)."""
    return [p.strip() for p in re.split(r"(?<=[.!?…])\s+|\n+", texte or "")
            if p.strip()]


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
            return (texte[:m.start()]).strip()
        vus.add(cle)
    return (texte or "").strip()


def _sans_copies(texte: str, deja_ecrit: str) -> str:
    """Retire du texte les phrases mot pour mot deja presentes ailleurs.

    C'est la garantie TECHNIQUE anti-recopie inter-parties : meme si le 1B
    essaie, la phrase copiee ne sort pas. Si tout est copie (texte vide),
    on garde l'original (mieux qu'une reponse vide).
    """
    if not texte or not deja_ecrit:
        return texte
    bas = deja_ecrit.lower()
    gardes = []
    for p in _phrases(texte):
        if len(p) >= _SEUIL_PHRASE and p.lower() in bas:
            continue                      # copie verbatim d'une partie precedente
        gardes.append(p)
    propre = " ".join(gardes).strip()
    return propre if len(propre) >= 40 else texte


def _sans_doublons(texte: str) -> str:
    """Retire les phrases repetees AU SEIN d'un meme texte.

    Utile pour les extraits web (deux resultats identiques colles cote a
    cote) et en garde-fou final avant livraison : une phrase longue deja
    dite n'a rien a faire deux fois dans la meme reponse.
    """
    if not texte:
        return texte
    vus: set[str] = set()
    gardes: list[str] = []
    for p in _phrases(texte):
        cle = p.lower()
        if len(p) >= _SEUIL_PHRASE and cle in vus:
            continue
        vus.add(cle)
        gardes.append(p)
    propre = " ".join(gardes).strip()
    return propre or texte


def _resume_sections(redige: list[str]) -> str:
    """Resume ANTI-COPIE des parties deja ecrites : titre + 1re phrase.

    L'ancienne version injectait les 900 DERNIERS CARACTERES bruts dans le
    prompt suivant — invitait litteralement le 1B a les recopier. Un resume
    court donne la coherence (ou en est-on ?) sans fournir le texte a copier.
    """
    lignes = []
    for bloc in redige:
        phrases = _phrases(bloc)
        tete = phrases[0] if phrases else bloc.strip()
        lignes.append(f"- {tete[:160]}")
    return "\n".join(lignes)


def _generer_avec_disjoncteur(llm, messages: list[dict], max_tokens: int,
                              echantillonnage: dict) -> str:
    """Generation STREAMEE avec coupure immediate sur boucle + 1 relance.

    - le stream est verifie tous les ~40 signes : des qu'une boucle nait,
      l'iteration s'arrete (le « kill » — plus aucun token de gaspille) ;
      l'appel suivant repart d'un KV-cache neuf (chaque appel est autonome).
    - relance UNIQUE : repeat_penalty 1.1 -> 1.25, temperature 0.5 -> 0.85.
    - echec des deux -> texte tronque avant la repetition.
    """
    texte, boucle = _streamer(llm, messages, max_tokens, echantillonnage)
    if not boucle:
        return texte
    LOG.warning("[disjoncteur] boucle detectee (%d signes) -> relance "
                "penalite 1.25 / temp 0.85", len(texte))
    relance = dict(echantillonnage)
    relance["temperature"] = 0.85
    relance["repeat_penalty"] = 1.25
    alt, boucle2 = _streamer(llm, messages, max(80, int(max_tokens * 0.8)),
                             relance)
    if alt and not boucle2:
        return alt
    return _couper_boucle(texte) or texte


def _streamer(llm, messages: list[dict], max_tokens: int,
              echantillonnage: dict) -> tuple[str, bool]:
    """(texte, boucle_detectee) — stream si possible, sinon appel classique."""
    try:
        flux = llm.create_chat_completion(messages=messages,
                                          max_tokens=max_tokens,
                                          stream=True, **echantillonnage)
    except TypeError:
        # signature sans support du stream : appel classique (post-check)
        flux = llm.create_chat_completion(messages=messages,
                                          max_tokens=max_tokens,
                                          **echantillonnage)
        texte = _contenu(flux)
        return texte, _en_boucle(texte)
    if isinstance(flux, dict):
        # « stream » ignore (version ancienne / faux llm de test) :
        # reponse complete d'un coup, meme protection en sortie
        texte = _contenu(flux)
        return texte, _en_boucle(texte)

    texte = ""
    prochain_controle = _SEUIL_PHRASE
    try:
        for morceau in flux:
            if isinstance(morceau, dict):
                choix = morceau.get("choices") or [{}]
                delta = (choix[0].get("delta") or {}).get("content") or ""
            else:
                delta = ""
            if not delta:
                continue
            texte += delta
            if len(texte) >= prochain_controle:
                prochain_controle = len(texte) + _SEUIL_PHRASE
                if _en_boucle(texte):
                    return _couper_boucle(texte), True
    except Exception as e:      # flux interrompu : on garde ce qui est bon
        LOG.warning("[disjoncteur] stream interrompu : %s", e)
    return texte.strip(), _en_boucle(texte)


def _contenu(sortie) -> str:
    """Extrait le texte d'une reponse (dict OpenAI-like)."""
    try:
        return ((sortie.get("choices") or [{}])[0].get("message", {})
                .get("content", "") or "").strip()
    except Exception:
        return ""

# ── Sorties contraintes (GBNF) ────────────────────────────────────────

# Le plan multi-pass DOIT etre complet et parsable : 3 parties + connecteurs.
# Le decodage contraint garantit ce format au niveau du token (100 % conforme,
# jamais tronque ni deforme) — la passe 2 recoit toujours des rails propres,
# meme quand le 1B derape. Alignement prompt/grammaire : _PROMPT_PLAN montre
# le format exact que la grammaire impose.
_GBNF_PLAN = (
    "root ::= partie{3} connecteurs\n"
    'partie ::= "PARTIE " [1-3] " : " ligne "\\n"\n'
    "ligne ::= [^\\n]{10,220}\n"
    'connecteurs ::= "CONNECTEURS : " ligne\n'
)

_grammaire_plan_cache = None


def grammaire_plan():
    """LlamaGrammar du plan multi-pass, ou None si GBNF indisponible.

    Compilee une seule fois puis mise en cache ; tout echec (build sans
    support grammaire, version ancienne) retombe proprement sur un plan
    libre = comportement d'avant.
    """
    global _grammaire_plan_cache
    if _grammaire_plan_cache is None:
        try:
            from llama_cpp import LlamaGrammar
            g = LlamaGrammar.from_string(_GBNF_PLAN)
            _grammaire_plan_cache = (True, g)
            LOG.info("[gbnf] grammaire plan chargee : sortie contrainte active")
        except Exception as e:
            LOG.info("[gbnf] grammaire indisponible (%s) -> plan libre", e)
            _grammaire_plan_cache = (False, None)
    return _grammaire_plan_cache[1]


def _couches_gpu() -> int:
    """Couches à offloader au GPU si le build llama.cpp le supporte.

    - build CUDA/Vulkan/RoceM + GPU présent -> -1 (tout sur GPU, vitesse max)
    - sinon -> 0 (CPU pur : c'est le cas de CE PC, Intel UHD = iGPU sans
      build GPU de llama.cpp, et c'est volontaire : CPU = 13,4 tok/s)
    La variable AURA_GPU=0 force le CPU même si un GPU est détecté.
    """
    if os.environ.get("AURA_GPU") == "0":
        return 0
    try:
        import importlib
        llama_cpp = importlib.import_module("llama_cpp")
        llama_cpp2 = importlib.import_module("llama_cpp.llama_cpp")
        # un build GPU expose ces fonctions C ; un build CPU pur, non
        if not (hasattr(llama_cpp2, "ggml_backend_cuda_init")
                or hasattr(llama_cpp2, "ggml_backend_vk_init")):
            return 0
        if os.name == "nt":
            import ctypes
            if not ctypes.windll.d3d12:
                return 0
            # présence d'un adaptateur non-cpu ? heuristique wmic légère
            try:
                import subprocess
                out = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    capture_output=True, text=True, timeout=5).stdout.lower()
            except Exception:
                out = ""
            gpu_presente = any(m in out for m in ("nvidia", "radeon", "arc",
                                                   "geforce", "rtx", "gtx"))
            return -1 if gpu_presente else 0
        return -1 if os.path.exists("/dev/dri") else 0
    except Exception:
        return 0


def llm_charge():
    """Llama charge, ou None si le modele n'est pas (encore) en memoire.

    Pour le hot-swap d'adaptateurs : il faut les attrs bas niveau
    (_model/_ctx) d'un Llama DEJA charge — jamais de chargement force
    ici (une question qui va vers le cerveau le chargera de toute facon).
    """
    return _llm


def _draft_model():
    """Speculative decoding in-process : prompt-lookup / n-gram.

    Le brouillon est extrait de l'historique lui-meme (recherche de n-grammes
    deja vus) — AUCUN second modele, donc aucun cout de RAM de poids.

    PAR DEFAUT OFF (AURA_SPEC=1 pour activer) et ce n'est pas un choix
    esthetique : llama_cpp force logits_all=True des qu'un draft_model est
    passe (~500 Mo de scores pour Llama-3.2-1B), et le bench mesure un
    NET RECUL sur ce CPU :
        AURA_SPEC=1 : 2.1 / 3.4 tok/s   (repétitif / neuf)
        AURA_SPEC=0 : 6.0 / 8.8 tok/s
    Le boosting reel passe par le chemin SERVEUR (--spec-type ngram-cache,
    voir serveur_lora._spec_type_ok : x2 sur texte repetitif, neutre ailleurs)
    qui n'a pas ce defaut. Ici l'interrupteur reste pour comparer/debuguer.

    Medusa / EAGLE-3 : impossibles sur Llama-3.2-1B (aucun checkpoint draft
    converti n'existe pour cette cible) — d'ou le prompt-lookup qui, lui,
    marche avec n'importe quel modele.
    """
    if os.environ.get("AURA_SPEC", "0") == "0":
        return None
    try:
        from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
        return LlamaPromptLookupDecoding(max_ngram_size=2, num_pred_tokens=10)
    except Exception as e:  # noqa: BLE001  (build sans llama_speculative)
        LOG.warning("Speculative decoding indisponible : %s", e)
        return None


def _charger():
    global _llm
    if _llm is not None:
        return _llm
    if not os.path.exists(_CHEMIN_GGUF):
        raise FileNotFoundError(
            f"GGUF introuvable : {_CHEMIN_GGUF}\n"
            "Telecharge-le avec : python scripts/telecharger_llama.py\n"
            "ou boote le .aef dont le cerveau est en cache : "
            "python -m aura.kernel aura_system.aef")

    from llama_cpp import Llama
    reg = _reglages_materiel()
    couches_gpu = _couches_gpu()
    t0 = time.time()
    _llm = Llama(
        model_path=_CHEMIN_GGUF,
        n_ctx=reg["n_ctx"],
        n_batch=reg["n_batch"],
        n_threads=reg["n_threads"],
        n_threads_batch=reg["n_threads_batch"],
        flash_attn=reg["flash_attn"],
        type_k=reg["type_kv"],          # KV cache q8_0 : RAM /2
        type_v=reg["type_kv"],
        n_gpu_layers=couches_gpu,       # GPU si dispo, sinon 0 (CPU)
        draft_model=_draft_model(),     # spec decoding : prompt-lookup (AURA_SPEC=0 -> None)
        verbose=False,
    )
    LOG.info("Llama 3.2 1B charge en %.1fs (%s, gpu_layers=%d, spec=%s)",
             time.time() - t0, os.path.basename(_CHEMIN_GGUF), couches_gpu,
             "on" if _llm.draft_model is not None else "off")
    return _llm


def _formater_contexte(contexte_web: str, formule: str) -> str:
    parties = []
    if contexte_web:
        parties.append(f"FAITS WEB (temps reel) :\n{contexte_web}")
    if formule:
        parties.append(f"FORMULE EXACTE (verifiee, erreur ~0) : Y = {formule}")
    return "\n\n".join(parties)


def _max_tokens_adaptatif(question: str) -> int:
    """Court pour les faits (rapide), long pour les explications (utile)."""
    q = question.lower()
    if any(m in q for m in ("explique", "raconte", "pourquoi", "compare",
                            "difference", "resume", "comment")):
        return 320
    return 120


# Cerveaux a reflexion native <think> : ils consomment des tokens de pensee
# AVANT la reponse — les budgets de tokens doivent en tenir compte.
_NOMS_THINK = ("minicpm", "deepseek-r1", "qwq", "qwen3", "thinking")


def _cerveau_think(llm) -> bool:
    """Vrai si le cerveau charge raisonne nativement en <think>...</think>.

    Deux signaux : nom de modele (minicpm, r1...) OU template jinja avec
    <think>/enable_thinking (interrupteur officiel : AURA_THINK=1 l'active
    globalement, think=True de generer() par question).
    """
    try:
        meta = getattr(llm, "metadata", {}) or {}
        tmpl = meta.get("tokenizer.chat_template", "") or ""
        if "<think>" in tmpl or "enable_thinking" in tmpl:
            return True
        nom = (str(getattr(llm, "model_path", "")) + " " +
               str(meta)).lower()
        return any(m in nom for m in _NOMS_THINK)
    except Exception:
        return False


def _avec_raisonnement(question: str) -> str:
    """Ajoute une consigne de decomposition pour les questions de logique."""
    ql = question.lower()
    if any(m in ql for m in _MOTS_LOGIQUE):
        return (question + "\n(Raisonne etape par etape, puis donne la reponse finale.)")
    return question


def _tours_fenetre(historique) -> list[dict]:
    """Fenetre glissante : accepte MemoireConversation (RLM) ou liste brute.

    MemoireConversation -> ses 4 derniers tours deja tronques ; liste
    brute (tests, usages directs) -> les 6 derniers tours comme avant.
    """
    if historique is None:
        return []
    fenetre = getattr(historique, "fenetre", None)
    if callable(fenetre):
        return fenetre()
    return list(historique)[-6:]


def generer(question: str, contexte_web: str = "", formule: str = "",
            max_tokens: int | None = None,
            historique: list[dict] | None = None,
            systeme: str | None = None,
            contexte_faits: str = "",
            think: bool | None = None,
            categorie: str = "",
            complexe: bool = False) -> str:
    """Genere une reponse avec Llama 3.2 1B.

    historique : liste de {'role': 'user'|'assistant', 'content': str} pour
    les conversations multi-tours (memoire courte, le modele se souvient).
    systeme : prompt systeme de remplacement (mode PoT du raisonneur).
    contexte_faits : triplets du graphe de faits VERIFIES (MiniRAG-lite).
    categorie : categorie routee (web/math/code/general) — pilote le
    hot-swap LoRA du serveur (AURA_SERVEUR=1) ou in-process (AURA_ADAPTATEURS=1).
    complexe : question a forte complexite -> Dynamic Compute (reflection
    masquee <thinking> dans le chemin serveur, budget tokens x2).
    """
    contexte = _formater_contexte(contexte_web, formule)
    # ── chemin 1 : serveur officiel llama-server (hot-swap HTTP fiable) ──
    try:
        from . import serveur_lora
        if serveur_lora._actifs():
            # l'orchestrateur applique deja le mix multi-LoRA (poids du
            # routeur) ; ici on n'active que si RIEN n'est applique
            # (appels directs a generer hors orchestrateur)
            if serveur_lora._actif is None:
                serveur_lora.activer(categorie)
            messages_srv = [{"role": "system",
                             "content": systeme or _SYSTEME}]
            for tour in _tours_fenetre(historique):
                messages_srv.append({"role": tour["role"],
                                     "content": tour["content"]})
            if contexte:
                messages_srv.append({"role": "user", "content": contexte})
                messages_srv.append({"role": "assistant",
                                     "content": "Compris, j'utilise uniquement ces faits."})
            if contexte_faits:
                messages_srv.append({"role": "user", "content":
                    f"BASE DE FAITS (verifiee) :\n{contexte_faits}"})
                messages_srv.append({"role": "assistant",
                                     "content": "Compris, je m'appuie sur ces faits verifies."})
            messages_srv.append({"role": "user", "content": question})
            # DYNAMIC COMPUTE : sur question complexe, un brouillon masque
            # <thinking> est demande (budget x2) puis separe par
            # separer_reflexion — l'utilisateur ne voit que la reponse.
            if complexe and not systeme:
                messages_srv.append({"role": "assistant", "content":
                    ("Pour les questions complexes, reflechis d'abord entre "
                     "<thinking> et </thinking> (analyse, plan, verification), "
                     "puis ecris la reponse finale apres la balise fermante.")})
                budget_srv = max((max_tokens or 0) * 2, 400)
            else:
                budget_srv = max_tokens or _max_tokens_adaptatif(question)
            reponse = serveur_lora.chat(messages_srv, max_tokens=budget_srv)
            if reponse:
                propre, _ = separer_reflexion(reponse)
                return propre or reponse
            LOG.info("[llama] serveur muet -> chemin in-process")
    except Exception as e:
        LOG.info("[llama] chemin serveur indisponible (%s) -> in-process", e)

    # ── chemin 2 : cerveau in-process (llama-cpp-python) ──
    try:
        llm = _charger()
    except Exception as e:
        LOG.error("[llama] chargement impossible : %s", e)
        return ("[Aura] Le cerveau Llama n'est pas disponible. "
                f"Details : {e}")

    messages = [{"role": "system", "content": systeme or _SYSTEME}]
    for tour in _tours_fenetre(historique):     # fenetre glissante ancree
        messages.append({"role": tour["role"], "content": tour["content"]})
    if contexte:
        messages.append({"role": "user", "content": contexte})
        messages.append({"role": "assistant",
                         "content": "Compris, j'utilise uniquement ces faits."})
    if contexte_faits:
        messages.append({"role": "user", "content":
            f"BASE DE FAITS (verifiee) :\n{contexte_faits}"})
        messages.append({"role": "assistant",
                         "content": "Compris, je m'appuie sur ces faits verifies."})
    # prompt systeme custom = expert special (PoT/logique/code) : il gere
    # ses propres instructions — le suffixe CoT generique le contredirait
    messages.append({"role": "user",
                     "content": question if systeme else _avec_raisonnement(question)})

    if max_tokens is None:
        max_tokens = _max_tokens_adaptatif(question)
    if _cerveau_think(llm):
        # cerveau a reflexion native : budget x2.5 (plancher 200) pour ne
        # pas couper la pensee avant la reponse
        max_tokens = max(int(max_tokens * 2.5), 200)

    # interrupteur de reflexion native : par defaut OFF (reponse directe,
    # rapide — l'orchestration d'Aura gere deja le raisonnement par experts).
    # AURA_THINK=1 ou think=True par question pour les puzzles difficiles.
    kwargs_template = {}
    want_think = think if think is not None else os.environ.get("AURA_THINK") == "1"
    if _cerveau_think(llm) and not want_think:
        # interrupteur officiel (templates type MiniCPM5/Qwen3) — supporté
        # par llama-cpp-python récent ; sur 0.3.35 il est ignoré, on gère
        # alors la pensée vide en formatant ChatML manuellement (voir plus bas)
        kwargs_template["enable_thinking"] = False

    try:
        t0 = time.time()
        try:
            sortie = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.4,      # faible variance : ancre sur les faits
                top_p=0.9,
                min_p=0.05,           # coupe la queue -> reponses plus sures
                repeat_penalty=1.1,
                stop=["<|eot_id|>", "\nUtilisateur:", "\nUser:"],
                **kwargs_template,
            )
        except (TypeError, ValueError):
            # version de llama-cpp-python sans kwargs de template : appel
            # standard (le strip de <think> + relance gerent le reste)
            sortie = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.4,
                top_p=0.9,
                min_p=0.05,
                repeat_penalty=1.1,
                stop=["<|eot_id|>", "\nUtilisateur:", "\nUser:"],
            )
        texte = (sortie.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        n_gen = sortie.get("usage", {}).get("completion_tokens", 0)
        if n_gen:
            duree = max(time.time() - t0, 1e-6)
            LOG.info("[llama] %d tokens en %.1fs = %.1f tok/s",
                     n_gen, duree, n_gen / duree)
        if texte:
            propre, reflexion = separer_reflexion(texte)
            if not propre and reflexion and not want_think:
                # pensee coupee, reponse vide malgre enable_thinking=False :
                # relance unique budget triple
                try:
                    sortie = llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max(max_tokens * 3, 512),
                        temperature=0.4, top_p=0.9, min_p=0.05,
                        repeat_penalty=1.1,
                        stop=["<|eot_id|>", "\nUtilisateur:", "\nUser:"],
                        **kwargs_template,
                    )
                    texte2 = (sortie.get("choices") or [{}])[0] \
                        .get("message", {}).get("content", "").strip()
                    if texte2:
                        propre, _ = separer_reflexion(texte2)
                except Exception as e:
                    LOG.warning("[llama] relance reflexion echouee : %s", e)
            # DISJONCTEUR ANTI-REPETITION : une boucle detectee dans la
            # reponse -> UNE relance (penalite 1.25 / temperature 0.85) ;
            # si elle derape aussi, on livre le texte COUPE avant la
            # repetition. La boucle ne sort jamais vers l'utilisateur.
            if propre and _en_boucle(propre):
                LOG.warning("[disjoncteur] boucle dans la reponse -> relance")
                try:
                    sortie2 = llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max(80, int(max_tokens * 0.8)),
                        temperature=0.85, top_p=0.9, min_p=0.05,
                        repeat_penalty=1.25,
                        stop=["<|eot_id|>", "\nUtilisateur:", "\nUser:"],
                        **kwargs_template,
                    )
                    alt, _ = separer_reflexion(_contenu(sortie2))
                    propre = alt if (alt and not _en_boucle(alt)) \
                        else _couper_boucle(propre)
                except Exception as e:      # jamais bloquant
                    LOG.warning("[disjoncteur] relance impossible : %s", e)
                    propre = _couper_boucle(propre)
            if propre:
                return propre
            # pensee non fermee (budget insuffisant) : ne JAMAIS livrer le
            # bloc <think> brut (le matching reponse/verite y serait faux)
            LOG.warning("[llama] reflexion non terminee (budget %d tokens)",
                        max_tokens)
            return ("[Aura] Reflexion interrompue avant la reponse. "
                    "Reformule plus court, ou relance avec AURA_THINK=1.")
        return texte or "[Aura] Reponse vide du modele."
    except Exception as e:
        LOG.error("[llama] generation echouee : %s", e)
        return f"[Aura] Erreur de generation : {e}"


def disponible() -> bool:
    """Le cerveau est utilisable : GGUF present ET llama_cpp importable.

    Le GGUF peut vivre dans le cache du kernel (.aef) sur une machine qui
    n'a pas le paquet lourd (venv CI, machine utilisateur) : dans ce cas
    les tests grandeur nature doivent se SAUTER, pas echouer.
    """
    if not os.path.exists(_CHEMIN_GGUF):
        return False
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


# ── Étape 2 : nettoyage du CoT ─────────────────────────────────────────

def separer_reflexion(reponse_brute: str) -> tuple[str, str]:
    """Extrait la reflexion <think> ou <thinking>...</think(ing)> et renvoie (propre, reflexion).

    Deux familles de balises : <thinking> (le CoT masque du mode riche) et
    <think> (la reflexion NATIVE des cerveaux type MiniCPM5/R1/QwQ). Dans les
    deux cas l'utilisateur ne voit que la reponse propre ; la reflexion part
    dans les logs (audit qualite) via LOG.debug.
    """
    reflexion = ""
    m = re.search(r"<think(?:ing)?>(.*?)</think(?:ing)?>", reponse_brute, re.DOTALL)
    if m:
        reflexion = m.group(1).strip()
        LOG.debug("[cot] reflexion : %s", reflexion[:400])
    propre = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", reponse_brute,
                    flags=re.DOTALL).strip()
    # balise fermante orpheline (generation coupee) : on coupe avant
    if re.search(r"<think(?:ing)?>", propre):
        propre = re.split(r"<think(?:ing)?>", propre)[0].strip()
    return propre, reflexion


# ── Étape 3 : generation multi-pass (plan -> redaction stylisee) ──────

def _generer_plan(llm, question: str, contexte: str) -> str:
    """Passe 1 : plan 3 parties + connecteurs, sortie CONTRAINTE par GBNF.

    La grammaire force aussi l'arret : le modele emet EOS des que 'root'
    est complete (max_tokens n'est qu'un garde-fou).
    """
    p1 = llm.create_chat_completion(
        messages=[{"role": "system", "content": _SYSTEME_PLAN},
                  {"role": "user", "content": _PROMPT_PLAN.format(
            contexte=contexte or "(aucun contexte fourni)",
            question=question)}],
        max_tokens=280, temperature=0.3,
        stop=["<|eot_id|>"],
        grammar=grammaire_plan(),
    )
    return (p1.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()


def generer_riche(question: str, contexte_web: str = "",
                  formule: str = "", historique: list[dict] | None = None) -> str:
    """Generation en 2 passes pour les reponses longues : plan puis style.

    Passe 1 : plan detaille + 5 connecteurs logiques (la feuille de route).
    Passe 2 : redaction stylisee guidee par le plan (le modele se concentre
    sur la forme, les rails sont deja poses).

    Repli : si une passe echoue, retour a la generation simple.
    """
    contexte = _formater_contexte(contexte_web, formule)
    contexte_court = contexte[:600]

    try:
        llm = _charger()
    except Exception as e:
        LOG.error("[llama] chargement impossible : %s", e)
        return generer(question, contexte_web, formule, historique=historique)

    # ── Passe 1 : plan + connecteurs (sortie contrainte par grammaire) ──
    t0_riche = time.time()
    try:
        plan = _generer_plan(llm, question, contexte)
        LOG.info("[multi-pass] plan en %.1fs (%d car)",
                 time.time() - t0_riche, len(plan))
    except Exception as e:
        LOG.warning("[multi-pass] passe 1 echouee : %s", e)
        plan = ""
    if not plan:
        return generer(question, contexte_web, formule, historique=historique)

    # ── Passe 2 : redaction SECTION PAR SECTION (inspiree de StoryWriter) ──
    # chaque section est une generation COURTE : le 1B reste dans sa zone
    # (peu de tokens = moins de derape) et la memoire des sections deja
    # ecrites garantit la coherence d'ensemble. Repli : passe unique.
    sections = [l for l in plan.splitlines()
                if l.strip().upper().startswith("PARTIE")]
    if len(sections) >= 2:
        redige: list[str] = []
        for num, titre in enumerate(sections, 1):
            # RESUME ANTI-COPIE (titre + 1re phrase) et non les 900
            # caracteres bruts precedents : c'etait la source directe de la
            # duplication partie 1 -> partie 2 observee en production.
            memoire = _resume_sections(redige)
            t0_sec = time.time()
            try:
                texte = _generer_avec_disjoncteur(
                    llm,
                    [{"role": "user", "content":
                        _PROMPT_SECTION.format(
                            numero=num, total=len(sections),
                            titre=titre.strip(), question=question,
                            role_section=_ROLES_SECTION.get(
                                min(num, 3), _ROLES_SECTION[3]),
                            contexte_court=contexte_court,
                            memoire=memoire or "(premiere section)")}],
                    190,
                    {"temperature": 0.5, "top_p": 0.95, "min_p": 0.05,
                     "repeat_penalty": 1.1, "stop": ["<|eot_id|>"]})
                # GARANTIE TECHNIQUE : une phrase mot pour mot deja ecrite
                # dans une partie precedente est retirees — impossible de
                # recopier le paragraphe precedent, meme si le 1B y veille.
                texte = _sans_copies(texte, "\n".join(redige))
            except Exception as e:
                LOG.warning("[storywriter] section %d echouee : %s", num, e)
                texte = ""
            if texte:
                redige.append(texte)
            LOG.info("[storywriter] section %d/%d en %.1fs (cumul %.1fs)",
                     num, len(sections), time.time() - t0_sec,
                     time.time() - t0_riche)
        if redige:
            return "\n\n".join(redige)
        LOG.warning("[storywriter] aucune section redigee -> passe unique")

    # ── repli : passe unique (comportement d'avant) ──
    try:
        messages = [{"role": "system", "content": _PROMPT_STYLE.format(
            plan=plan, question=question, contexte_court=contexte_court)}]
        for tour in _tours_fenetre(historique):
            messages.append({"role": tour["role"], "content": tour["content"]})
        p2 = llm.create_chat_completion(
            messages=messages, max_tokens=340, temperature=0.5,
            top_p=0.95, min_p=0.05, repeat_penalty=1.1,
            stop=["<|eot_id|>"])
        redaction = _contenu(p2)
        if _en_boucle(redaction):
            LOG.warning("[disjoncteur] boucle dans la passe style -> troncature")
            redaction = _couper_boucle(redaction)
    except Exception as e:
        LOG.warning("[multi-pass] passe 2 echouee : %s", e)
        return generer(question, contexte_web, formule, historique=historique)

    if not redaction:
        return generer(question, contexte_web, formule, historique=historique)
    return redaction
