"""Agents Google ADK sur le cerveau local (opt-in : AURA_ADK=1).

Planificateur puis relecteur, pour les taches complexes seulement. Le modele
reste local : ADK appelle `llama_cerveau.generer` via un adaptateur BaseLlm.
"""
import asyncio
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import agents

LOG = logging.getLogger("aura.agents_adk")
APP = "aura_plan"
_ETAPE = re.compile(r"^\s*\d[\.\)]\s+\S", re.MULTILINE)
_RELECTURE = ("Relis ce plan brouillon et renvoie UNIQUEMENT le plan final : "
              "2 a 3 etapes au format '1. ...' '2. ...', sans autre texte.\n\n"
              "PLAN BROUILLON :\n{plan}")


def actif() -> bool:
    return agents.actifs() and os.environ.get("AURA_ADK", "0") == "1"


def _texte(contenu) -> str:
    if contenu is None:
        return ""
    if isinstance(contenu, str):
        return contenu
    return "".join(p.text or "" for p in contenu.parts or [])


def _chaine():
    """Import paresseux : google-adk reste optionnel."""
    from google.adk.agents import LlmAgent, SequentialAgent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class ModeleLocal(BaseLlm):
        model: str = "aura-local"

        async def generate_content_async(self, llm_request, stream=False):
            from . import llama_cerveau
            msgs = [{"role": "assistant" if c.role == "model" else "user",
                     "content": _texte(c)} for c in llm_request.contents]
            if not msgs:
                return
            systeme = _texte(llm_request.config.system_instruction) or None
            # generer() est bloquant : hors de la boucle asyncio.
            texte = await asyncio.to_thread(
                llama_cerveau.generer, msgs[-1]["content"], systeme=systeme,
                historique=msgs[:-1] or None, max_tokens=160)
            yield LlmResponse(content=types.Content(
                role="model", parts=[types.Part(text=texte or "")]))

    return SequentialAgent(name="pipeline_plan", sub_agents=[
        LlmAgent(name="planificateur", model=ModeleLocal(),
                 instruction=agents._SYSTEME_PLAN, output_key="plan"),
        LlmAgent(name="relecteur", model=ModeleLocal(),
                 instruction=_RELECTURE, output_key="plan_final"),
    ])


async def _executer(question: str) -> str:
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=_chaine(), app_name=APP)
    sid = uuid.uuid4().hex
    await runner.session_service.create_session(
        app_name=APP, user_id="aura", session_id=sid)
    message = types.Content(role="user", parts=[types.Part(text=question)])
    async for _ in runner.run_async(user_id="aura", session_id=sid,
                                    new_message=message):
        pass
    session = await runner.session_service.get_session(
        app_name=APP, user_id="aura", session_id=sid)
    return _texte_etat(session.state)


def _texte_etat(etat) -> str:
    return str((etat or {}).get("plan_final", "") or "")


def planifier_adk(question: str) -> str | None:
    """Plan 2-3 etapes via ADK, ou None (le pipeline repasse par agents.planifier)."""
    if not actif() or not agents.est_tache_complexe(question):
        return None
    try:
        # asyncio.run refuse une boucle deja active : on passe par un thread.
        with ThreadPoolExecutor(max_workers=1) as pool:
            brut = pool.submit(lambda: asyncio.run(_executer(question))).result()
    except Exception as e:
        LOG.info("[adk] chaine impossible (%s) -> repli", e)
        return None
    if not 2 <= len(_ETAPE.findall(brut)) <= 3:
        LOG.info("[adk] plan inexploitable")
        return None
    return brut.strip()
