"""API locale HTTP : le coeur RontoGPT expose /v1/* pour la GUI, le vocal, les scripts.

Le but de l'industrialisation : UN seul process porte le cerveau, et tous les
front-ends (gui_web, RontoGPT-Vocal, scripts, tierces) lui parlent par JSON au
lieu de charger chacun leur propre Llama. Standard decompat : stdlib only
(aucune dependance lourde de plus), boucle d'evenements served en threads.

Demarrage :
    python -m aura.api               # port AURA_API_PORT (8760), preload du cerveau
    python -m aura.api --no-preload  # cerveau charge a la 1re question
    python -m aura.api --smoke       # auto-test puis exit 0 (CI / verification)

Routes :
    GET  /v1/health  -> {"status": "ok", "version": "..."}
    GET  /v1/status  -> etat du coeur (prete / chargement / erreur / serveur / gpu)
    GET  /v1/config  -> configuration effective (AURA_*)
    GET  /v1/coucou  -> le pont vers l'ile Coucou est-il disponible ?
    POST /v1/ask     {"question": "..."} -> {ok, reponse, experts, cerveau, latence,
                     en_attente, clarification}
    POST /v1/ask/stream {"question": "..."} -> SSE : data {"type":"morceau",
                     texte, phase} ou phase = "pensee" | "reponse" (la
                     reflexion ET la reponse sont poussees, chacune avec sa
                     phase : le client les affiche comme il veut)
                     puis data {"type":"final", reponse, experts, latence}
                     puis data [DONE]

`en_attente` = la reponse n'est PAS une reponse : la porte de clarification
(aura/clarification.py) a juge la question trop ouverte et demande une
precision. Le tour SUIVANT porte la precision ; le client n'a rien a faire
de special (c'est le meme couple question/reponse), mais il peut afficher
un etat distinct (« question en attente ») et `clarification` en donne la
raison (`famille`, `manque`, `questions`).
    POST /v1/notify  {"message": "...", "evenement": "termine"} -> {ok, envoye}
                     (pousse un evenement vers l'ile Coucou / Mochi)
    POST /v1/learn   {"sujet","relation","objet"} -> edite un FAIT verifie
                     ou {"question","correcte"} -> enregistre une correction
                     (edition de connaissance SANS reentrainement)

Securite : bind 127.0.0.1 par defaut (jamais expose sur le reseau). Si
AURA_API_CLE est posee, chaque appel doit fournir l'entete X-RontoGPT-Cle.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config as _config

_config.appliquer()

_LOG_COURT = 60_000          # corps de requete acceptes (60 Ko)

# Le flux de tokens est GLOBAL (un seul cerveau) : une seule requete streamee
# a la fois, sinon les morceaux de deux questions se melangeraient.
_FLUX_LIBRE = threading.Lock()


def _reponse_detaille(ia, question: str) -> dict:
    """Le resultat complet d'une question, au format de /v1/ask."""
    t0 = time.time()
    r = ia.executer_detaille(question)
    return {
        "ok": True,
        "reponse": str(r.get("reponse", "")),
        "experts": sorted(r.get("experts") or set()),
        "cerveau": str(r.get("cerveau_choisi", "-")),
        "latence": round(time.time() - t0, 2),
        # PORTE DE CLARIFICATION : la reponse est une QUESTION de relance.
        # Un client qui l'ignore fonctionne (c'est du texte), mais un client
        # qui la lit peut marquer l'etat au lieu de presenter une relance
        # comme une reponse.
        "en_attente": bool(r.get("en_attente")),
        "clarification": r.get("clarification"),
    }


class _Etat:
    """Coeur partage : IA lazy (un seul chargement), thread-safe."""

    def __init__(self, smoke: bool = False):
        self.smoke = smoke
        self.ia = None
        self.erreur: str | None = None
        self.chargement = False
        self.verrou = threading.Lock()

    def charger(self):
        """Charge RontoGPT une seule fois (le premier appel attend)."""
        if self.smoke:
            self.erreur = "mode smoke : cerveau non charge"
            return None
        with self.verrou:
            if self.ia is not None or self.erreur is not None:
                return self.ia
            self.chargement = True
            try:
                from .orchestrateur import RontoGPT
                self.ia = RontoGPT()
            except Exception as e:               # jamais de 500 silencieux
                self.erreur = str(e)
            finally:
                self.chargement = False
            return self.ia


def _verifier_cle(entetes) -> bool:
    attendue = os.environ.get("AURA_API_CLE")
    return (not attendue or entetes.get("X-RontoGPT-Cle") == attendue
            or entetes.get("X-Aura-Cle") == attendue)


class _Handler(BaseHTTPRequestHandler):
    etat: _Etat = _Etat()          # remplace par _creer_handler()

    # -- utilitaires --------------------------------------------------
    def log_message(self, format, *args):        # journalisation calmee
        pass

    def _json(self, donnees, code: int = 200) -> None:
        corps = json.dumps(donnees, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def _corps(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > _LOG_COURT:
                return {}
            return json.loads(self.rfile.read(n).decode() or "{}")
        except Exception:
            return {}

    # -- GET ----------------------------------------------------------
    def do_GET(self):                                # noqa: N802 (stdlib)
        if not _verifier_cle(self.headers):
            return self._json({"ok": False, "erreur": "cle invalide"}, 401)
        if self.path == "/v1/health":
            return self._json({"status": "ok",
                               "version": _config.version()})
        if self.path == "/v1/status":
            from . import serveur_lora
            e = self.etat
            return self._json({
                "version": _config.version(),
                "profil": os.environ.get("AURA_PROFIL", "equilibre"),
                "prete": e.ia is not None,
                "chargement": e.chargement,
                "erreur": e.erreur,
                "serveur_llama": serveur_lora._actifs(),
                "gpu": os.environ.get("AURA_GPU", "auto"),
                "budget_s": os.environ.get("AURA_BUDGET_S", "45"),
            })
        if self.path == "/v1/config":
            return self._json({"version": _config.version(),
                               "environnement": _config.efficace()})
        if self.path == "/v1/coucou":
            from . import coucou
            return self._json({
                "ok": True,
                "disponible": coucou.est_disponible(),
                "evenements": list(coucou.EVENEMENTS),
            })
        return self._json({"ok": False, "erreur": "route inconnue"}, 404)

    # -- POST ---------------------------------------------------------
    def do_POST(self):                               # noqa: N802 (stdlib)
        if not _verifier_cle(self.headers):
            return self._json({"ok": False, "erreur": "cle invalide"}, 401)
        if self.path == "/v1/notify":
            return self._notify()
        if self.path == "/v1/learn":
            return self._learn()
        if self.path == "/v1/ask/stream":
            return self._ask_stream()
        if self.path != "/v1/ask":
            return self._json({"ok": False, "erreur": "route inconnue"}, 404)
        return self._ask()

    def _ask(self) -> None:
        question = str(self._corps().get("question") or "").strip()
        if not question:
            return self._json({"ok": False,
                               "erreur": "champ 'question' requis"}, 400)
        ia = self.etat.ia or self.etat.charger()
        if ia is None:
            return self._json({"ok": False,
                               "erreur": self.etat.erreur or "IA non chargee"},
                              503)
        try:
            return self._json(_reponse_detaille(ia, question))
        except Exception as e:                       # l'IA vit encore
            return self._json({"ok": False, "erreur": str(e)}, 500)

    # -- streaming SSE : les tokens partent au fil de l'eau -----------
    def _sse(self, donnees: dict) -> None:
        """Un evenement Server-Sent Events ; un client parti n'est pas une erreur."""
        try:
            charge = json.dumps(donnees, ensure_ascii=False)
            self.wfile.write(f"data: {charge}\n\n".encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _ask_stream(self) -> None:
        corps = self._corps()
        question = str(corps.get("question") or "").strip()
        if not question:
            return self._json({"ok": False,
                               "erreur": "champ 'question' requis"}, 400)
        ia = self.etat.ia or self.etat.charger()
        if ia is None:
            return self._json({"ok": False,
                               "erreur": self.etat.erreur or "IA non chargee"},
                              503)
        if not _FLUX_LIBRE.acquire(blocking=False):
            return self._json(
                {"ok": False, "erreur": "un flux est deja en cours — reessaie"},
                409)

        from . import llama_cerveau
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        # Le generateur publie chaque morceau depuis CE thread : l'abonne
        # ecrit donc directement dans la reponse, sans file intermediaire.
        # LES DEUX PHASES, et chacune ANNONCEE : c'est la demande produit
        # « on voit la pensee ET la reponse ». Le client recoit son `phase`
        # et peut les afficher differemment ; un client qui ignore le champ
        # se contente de concatener, comme avant.
        desabonner = llama_cerveau.abonner_flux_phase(
            lambda texte, phase: self._sse({"type": "morceau",
                                            "texte": texte, "phase": phase}))
        try:
            resultat = _reponse_detaille(ia, question)
            self._sse({"type": "final", **resultat})
        except Exception as e:                       # noqa: BLE001
            self._sse({"type": "erreur", "erreur": str(e)})
        finally:
            desabonner()
            _FLUX_LIBRE.release()
            self._sse({"type": "fin"})
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    # -- edition de connaissance (/v1/learn) --------------------------
    def _learn(self) -> None:
        """Edite la connaissance SANS reentrainement (cf. note MEMIT plus bas)."""
        corps = self._corps()
        sujet = str(corps.get("sujet") or "").strip()
        relation = str(corps.get("relation") or "").strip()
        objet = str(corps.get("objet") or "").strip()
        if sujet and relation and objet:
            from . import graphe_faits
            ajoute = graphe_faits.ajouter(
                sujet, relation, objet, str(corps.get("source") or "api"))
            return self._json({"ok": True, "type": "fait",
                               "ajoute": bool(ajoute)})
        question = str(corps.get("question") or "").strip()
        correcte = str(corps.get("correcte") or "").strip()
        if question and correcte:
            from . import autoamelioration
            autoamelioration.enregistrer_correction(
                question, str(corps.get("mauvaise") or ""), correcte,
                str(corps.get("categorie") or "general"))
            return self._json({"ok": True, "type": "correction",
                               "enregistre": True})
        return self._json(
            {"ok": False,
             "erreur": "faut (sujet, relation, objet) ou (question, correcte)"},
            400)

    # -- pont Coucou (ile Mochi) --------------------------------------
    def _notify(self) -> None:
        corps = self._corps()
        message = str(corps.get("message") or "").strip()
        if not message:
            return self._json({"ok": False,
                               "erreur": "champ 'message' requis"}, 400)
        evenement = str(corps.get("evenement") or "notification")
        agent = str(corps.get("agent") or "aura")
        from . import coucou
        try:
            resultat = coucou.notifier(message, evenement=evenement, agent=agent)
        except ValueError as e:                      # evenement mal nomme
            return self._json({"ok": False, "erreur": str(e)}, 400)
        return self._json({"ok": True, **resultat})


def _creer_handler(etat: _Etat):
    return type("HandlerAvecEtat", (_Handler,), {"etat": etat})


# -- auto-test (CI, verification d'installation) ------------------------
def _smoke(port: int) -> int:
    base = f"http://127.0.0.1:{port}"

    def _get(route: str, attendu: int = 200) -> tuple[int, dict]:
        try:
            with urllib.request.urlopen(base + route, timeout=10) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "{}")

    def _post(route: str, corps: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            base + route, data=json.dumps(corps).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "{}")

    checks: list[tuple[str, bool]] = []
    checks.append(("health", _get("/v1/health")[1].get("status") == "ok"))
    checks.append(("status : non prete", _get("/v1/status")[1].get("prete") is False))
    checks.append(("config", "AURA_PROFIL" in _get("/v1/config")[1]["environnement"]))
    checks.append(("question vide -> 400", _post("/v1/ask", {"question": ""})[0] == 400))
    code, rep = _post("/v1/ask", {"question": "bonjour"})
    checks.append(("smoke sans cerveau -> 503 ok:false",
                   code == 503 and rep.get("ok") is False))
    checks.append(("route inconnue -> 404", _get("/v1/inconnu")[0] == 404))
    ok = all(v for _, v in checks)
    for nom, v in checks:
        print(f"  [{'OK' if v else 'KO'}] {nom}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aura.api")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("AURA_API_PORT", "8760")))
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 par defaut : ne jamais exposer sur le reseau")
    ap.add_argument("--no-preload", action="store_true",
                    help="ne pas charger le cerveau au demarrage")
    ap.add_argument("--smoke", action="store_true",
                    help="auto-test des routes puis exit 0")
    args = ap.parse_args(argv)

    etat = _Etat(smoke=args.smoke)
    serveur = ThreadingHTTPServer((args.host, args.port), _creer_handler(etat))
    serveur.daemon_threads = True        # un client lent ne bloque pas l'arret
    fil = threading.Thread(target=serveur.serve_forever, daemon=True)
    fil.start()
    print(f"[api] http://{args.host}:{args.port}/v1/health "
          f"(profil {os.environ.get('AURA_PROFIL', 'equilibre')})")

    try:
        if args.smoke:
            code = _smoke(args.port)
            return code
        # rechauffage du serveur Vulkan en tache de fond : la premiere
        # question ne paie pas les 5-15 s de boot llama-server
        try:
            from . import serveur_lora
            serveur_lora.rechauffer()
        except Exception:              # jamais bloquant
            pass
        if not args.no_preload:
            threading.Thread(target=etat.charger, daemon=True).start()
            print("[api] chargement du cerveau en tache de fond...")
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[api] arret")
    finally:
        serveur.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
