"""Cerveau Llama 3.2 1B Instruct (Q4_K_M) — remplace Mamba + RWKV.

Pourquoi ce choix :
- **Instruction-tuned** : contrairement a Mamba 790M et RWKV 430M (base models
  qui completent du texte), Llama 3.2 1B a ete affine pour suivre des
  instructions → bon francais ET bon anglais, sans fine-tuning.
- **Q4_K_M** : quantification 4 bits (~807 Mo) → tient en RAM, rapide sur CPU.
- **llama.cpp** : moteur C++ optimise (AVX2, threads), bien plus rapide que
  PyTorch pur pour les petits modeles.

Interface identique aux anciens cerveaux : `generer(question, contexte_web, formule)`.
"""
import logging
import os
import time

LOG = logging.getLogger("aura.llama")

_CHEMIN_GGUF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "modeles", "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
)

_SYSTEME = (
    "Tu es Aura, une IA francophone et anglophone precise et concise. "
    "Reponds en francais si la question est en francais, en anglais sinon. "
    "2 phrases maximum sauf demande explicite. Utilise uniquement les faits "
    "fournis ci-dessous s'ils sont presents ; si tu ne sais pas, dis-le."
)

# Parsing Llama 3.2 (format chat officiel, applique par llama.cpp via le
# template embarque dans le GGUF).

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
    t0 = time.time()
    _llm = Llama(
        model_path=_CHEMIN_GGUF,
        n_ctx=2048,            # contexte court = RAM faible + prefill rapide
        n_batch=256,
        n_threads=max(1, (os.cpu_count() or 4) // 2),  # coeurs physiques
        n_threads_batch=max(1, os.cpu_count() or 4),
        verbose=False,
    )
    LOG.info("Llama 3.2 1B charge en %.1fs", time.time() - t0)
    return _llm


def _formater_contexte(contexte_web: str, formule: str) -> str:
    parties = []
    if contexte_web:
        parties.append(f"FAITS WEB (temps reel) :\n{contexte_web}")
    if formule:
        parties.append(f"FORMULE EXACTE (verifiee, erreur ~0) : Y = {formule}")
    return "\n\n".join(parties)


def generer(question: str, contexte_web: str = "", formule: str = "",
            max_tokens: int = 220) -> str:
    """Genere une reponse avec Llama 3.2 1B (instruction-tuned, FR+EN)."""
    try:
        llm = _charger()
    except Exception as e:
        LOG.error("[llama] chargement impossible : %s", e)
        return ("[Aura] Le cerveau Llama n'est pas disponible. "
                f"Details : {e}")

    contexte = _formater_contexte(contexte_web, formule)
    messages = [{"role": "system", "content": _SYSTEME}]
    if contexte:
        messages.append({"role": "user", "content": contexte})
        messages.append({"role": "assistant",
                         "content": "Compris, j'utilise uniquement ces faits."})
    messages.append({"role": "user", "content": question})

    try:
        import torch  # noqa: F401 — inutile ici, present pour coherence CPU
    except ImportError:
        pass

    try:
        t0 = time.time()
        sortie = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.4,      # faible variance : ancre sur les faits
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|eot_id|>", "\nUtilisateur:", "\nUser:"],
        )
        texte = (sortie.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        n_gen = sortie.get("usage", {}).get("completion_tokens", 0)
        if n_gen:
            duree = max(time.time() - t0, 1e-6)
            LOG.info("[llama] %d tokens en %.1fs = %.1f tok/s", n_gen, duree, n_gen / duree)
        return texte or "[Aura] Reponse vide du modele."
    except Exception as e:
        LOG.error("[llama] generation echouee : %s", e)
        return f"[Aura] Erreur de generation : {e}"


def disponible() -> bool:
    return os.path.exists(_CHEMIN_GGUF)
