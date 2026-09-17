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
_CHEMINS = {
    "Q4_K_M": os.path.join(_DOSSIER, "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "Q6_K": os.path.join(_DOSSIER, "Llama-3.2-1B-Instruct-Q6_K.gguf"),
}
_CHEMIN_GGUF = _CHEMINS[os.environ.get("AURA_GGUF", "Q4_K_M").upper()]

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

# ── Étape 2 : Chain-of-Thought masque ──────────────────────────────────

_COT_SYSTEME = (
    "Tu es Aura-1B, un systeme d'IA de niveau expert. Avant de repondre a "
    "l'utilisateur, tu dois OBLIGATOIREMENT mener une reflexion approfondie "
    "pas a pas. Ta reponse doit suivre strictement cette structure :\n"
    "<thinking>\n"
    "1. Analyse de la demande : de quoi s'agit-il precisement ?\n"
    "2. Strategie : quel ton, quels mots-cles du contexte integrer ?\n"
    "3. Plan : redige le plan de la reponse en 3 sections maximum.\n"
    "</thinking>\n"
    "[Ta reponse finale redigee avec un style riche et soutenu commence ici, "
    "hors des balises]"
)

# ── Étape 3 : prompts de la generation multi-pass ─────────────────────

_PROMPT_PLAN = (
    "En te basant sur ces faits :\n{contexte}\n\n"
    "et sur cette question : {question}\n\n"
    "Genere UNIQUEMENT un plan ultra-detaille en 3 parties et liste 5 "
    "connecteurs logiques avances (ex : 'Neanmoins', 'Par consequent') "
    "que tu utiliseras."
)

_PROMPT_STYLE = (
    "Tu es un redacteur litteraire et scientifique de haut niveau.\n"
    "En te basant sur ce plan :\n{plan}\n\n"
    "Redige l'analyse finale de la question : {question}\n\n"
    "CONSIGNES DE STYLE STRICTES :\n"
    "- Utilise un vocabulaire riche, precis et varie (evite les mots valises "
    "comme 'faire', 'dire', 'chose').\n"
    "- Fais des phrases courtes mais percutantes, reliees par les connecteurs "
    "logiques listes.\n"
    "- Interdiction de repeter la meme structure de phrase.\n"
    "- Integre naturellement les faits suivants : {contexte_court}\n"
    "- Reponds en francais, sans balises ni commentaires."
)

_llm = None


def _charger():
    global _llm
    if _llm is not None:
        return _llm
    if not os.path.exists(_CHEMIN_GGUF):
        raise FileNotFoundError(
            f"GGUF introuvable : {_CHEMIN_GGUF}\n"
            "Telecharge-le avec : python scripts/telecharger_llama.py")

    from llama_cpp import Llama
    reg = _reglages_materiel()
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
        verbose=False,
    )
    LOG.info("Llama 3.2 1B charge en %.1fs (%s)", time.time() - t0,
             os.path.basename(_CHEMIN_GGUF))
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


def _avec_raisonnement(question: str) -> str:
    """Ajoute une consigne de decomposition pour les questions de logique."""
    ql = question.lower()
    if any(m in ql for m in _MOTS_LOGIQUE):
        return (question + "\n(Raisonne etape par etape, puis donne la reponse finale.)")
    return question


def generer(question: str, contexte_web: str = "", formule: str = "",
            max_tokens: int | None = None,
            historique: list[dict] | None = None) -> str:
    """Genere une reponse avec Llama 3.2 1B.

    historique : liste de {'role': 'user'|'assistant', 'content': str} pour
    les conversations multi-tours (memoire courte, le modele se souvient).
    """
    try:
        llm = _charger()
    except Exception as e:
        LOG.error("[llama] chargement impossible : %s", e)
        return ("[Aura] Le cerveau Llama n'est pas disponible. "
                f"Details : {e}")

    contexte = _formater_contexte(contexte_web, formule)
    messages = [{"role": "system", "content": _SYSTEME}]
    for tour in (historique or [])[-6:]:          # memoire : 6 derniers tours
        messages.append({"role": tour["role"], "content": tour["content"]})
    if contexte:
        messages.append({"role": "user", "content": contexte})
        messages.append({"role": "assistant",
                         "content": "Compris, j'utilise uniquement ces faits."})
    messages.append({"role": "user", "content": _avec_raisonnement(question)})

    if max_tokens is None:
        max_tokens = _max_tokens_adaptatif(question)

    try:
        t0 = time.time()
        sortie = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.4,      # faible variance : ancre sur les faits
            top_p=0.9,
            min_p=0.05,           # coupe la queue -> reponses plus sures
            repeat_penalty=1.1,
            stop=["<|eot_id|>", "\nUtilisateur:", "\nUser:"],
        )
        texte = (sortie.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        n_gen = sortie.get("usage", {}).get("completion_tokens", 0)
        if n_gen:
            duree = max(time.time() - t0, 1e-6)
            LOG.info("[llama] %d tokens en %.1fs = %.1f tok/s",
                     n_gen, duree, n_gen / duree)
        return texte or "[Aura] Reponse vide du modele."
    except Exception as e:
        LOG.error("[llama] generation echouee : %s", e)
        return f"[Aura] Erreur de generation : {e}"


def disponible() -> bool:
    return os.path.exists(_CHEMIN_GGUF)


# ── Étape 2 : nettoyage du CoT ─────────────────────────────────────────

def separer_reflexion(reponse_brute: str) -> tuple[str, str]:
    """Extrait la reflexion <thinking>...</thinking> et renvoie (propre, reflexion).

    L'utilisateur ne voit que la reponse propre ; la reflexion part dans les
    logs (audit qualite) via LOG.debug.
    """
    reflexion = ""
    m = re.search(r"<thinking>(.*?)</thinking>", reponse_brute, re.DOTALL)
    if m:
        reflexion = m.group(1).strip()
        LOG.debug("[cot] reflexion : %s", reflexion[:400])
    propre = re.sub(r"<thinking>.*?</thinking>", "", reponse_brute,
                    flags=re.DOTALL).strip()
    # balise fermante orpheline (generation coupee) : on coupe avant
    if "<thinking>" in propre and "</thinking>" not in propre:
        propre = propre.split("<thinking>")[0].strip()
    return propre, reflexion


# ── Étape 3 : generation multi-pass (plan -> redaction stylisee) ──────

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

    # ── Passe 1 : plan + connecteurs ──
    try:
        p1 = llm.create_chat_completion(
            messages=[{"role": "user", "content": _PROMPT_PLAN.format(
                contexte=contexte or "(aucun contexte fourni)",
                question=question)}],
            max_tokens=220, temperature=0.3,
            stop=["<|eot_id|>"])
        plan = (p1.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        LOG.warning("[multi-pass] passe 1 echouee : %s", e)
        plan = ""
    if not plan:
        return generer(question, contexte_web, formule, historique=historique)

    # ── Passe 2 : redaction stylisee ──
    try:
        messages = [{"role": "system", "content": _PROMPT_STYLE.format(
            plan=plan, question=question, contexte_court=contexte_court)}]
        for tour in (historique or [])[-4:]:
            messages.append({"role": tour["role"], "content": tour["content"]})
        p2 = llm.create_chat_completion(
            messages=messages, max_tokens=380, temperature=0.5,
            top_p=0.95, min_p=0.05, repeat_penalty=1.1,
            stop=["<|eot_id|>"])
        redaction = (p2.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        LOG.warning("[multi-pass] passe 2 echouee : %s", e)
        return generer(question, contexte_web, formule, historique=historique)

    if not redaction:
        return generer(question, contexte_web, formule, historique=historique)
    return redaction
