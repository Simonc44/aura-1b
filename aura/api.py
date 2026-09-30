"""API locale HTTP : le coeur Aura expose /v1/* pour la GUI, le vocal, les scripts.

Le but de l'industrialisation : UN seul process porte le cerveau, et tous les
front-ends (gui_web, Aura-Vocal, scripts, tierces) lui parlent par JSON au
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
    POST /v1/ask     {"question": "..."} -> {ok, reponse, experts, cerveau, latence}

Securite : bind 127.0.0.1 par defaut (jamais expose sur le reseau). Si
AURA_API_CLE est posee, chaque appel doit fournir l'entete X-Aura-Cle.
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


class _Etat:
    """Coeur partage : IA lazy (un seul chargement), thread-safe."""

    def __init__(self, smoke: bool = False):
        self.smoke = smoke
        self.ia = None
        self.erreur: str | None = None
        self.chargement = False
        self.verrou = threading.Lock()

    def charger(self):
        """Charge Aura1B une seule fois (le premier appel attend)."""
        if self.smoke:
            self.erreur = "mode smoke : cerveau non charge"
            return None
        with self.verrou:
            if self.ia is not None or self.erreur is not None:
                return self.ia
            self.chargement = True
            try:
                from .orchestrateur import Aura1B
                self.ia = Aura1B()
            except Exception as e:               # jamais de 500 silencieux
                self.erreur = str(e)
            finally:
                self.chargement = False
            return self.ia


def _verifier_cle(entetes) -> bool:
    attendue = os.environ.get("AURA_API_CLE")
    return not attendue or entetes.get("X-Aura-Cle") == attendue


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
        return self._json({"ok": False, "erreur": "route inconnue"}, 404)

    # -- POST ---------------------------------------------------------
    def do_POST(self):                               # noqa: N802 (stdlib)
        if not _verifier_cle(self.headers):
            return self._json({"ok": False, "erreur": "cle invalide"}, 401)
        if self.path != "/v1/ask":
            return self._json({"ok": False, "erreur": "route inconnue"}, 404)
        question = str(self._corps().get("question") or "").strip()
        if not question:
            return self._json({"ok": False,
                               "erreur": "champ 'question' requis"}, 400)
        ia = self.etat.ia or self.etat.charger()
        if ia is None:
            return self._json({"ok": False,
                               "erreur": self.etat.erreur or "IA non chargee"},
                              503)
        t0 = time.time()
        try:
            r = ia.executer_detaille(question)
            return self._json({
                "ok": True,
                "reponse": str(r.get("reponse", "")),
                "experts": sorted(r.get("experts") or set()),
                "cerveau": str(r.get("cerveau_choisi", "-")),
                "latence": round(time.time() - t0, 2),
            })
        except Exception as e:                       # l'IA vit encore
            return self._json({"ok": False, "erreur": str(e)}, 500)


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
