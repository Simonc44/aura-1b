"""Supervision de l'API locale : un seul cerveau, relance sur plantage.

Ce module est l'ancien `main.py` de la racine, FUSIONNE dans le point d'entree
unique du produit (`python -m aura --superviser` / `rontogpt --superviser`) :
il n'y a plus qu'un lanceur a connaitre, et il vit avec le reste du code.

- `--superviser`            : l'API tourne SOUS SURVEILLANCE (sante verifiee,
                              relance avec delai croissant si elle plante)
                              pendant que le TUI s'y connecte ;
- `--superviser --api`      : API seule, sans interface (l'ancien
                              `python main.py --api`).

L'API vit dans un PROCESSUS SEPARE (`python -m aura.api`) : c'est ce qui permet
de la relancer sans perdre l'interface. `AURA_API_PORT` choisit le port (8760),
et la sante se lit sur `/v1/health` — une API deja lancee ailleurs est
REUTILISEE telle quelle plutot que doublee.
"""
import logging
import os
import subprocess
import sys
import time
import urllib.request

LOG = logging.getLogger("aura.superviseur")

PORT = int(os.environ.get("AURA_API_PORT", "8760"))
URL_SANTE = f"http://127.0.0.1:{PORT}/v1/health"
MAX_RELANCES = 5
DELAI_SANTE_S = 60.0
PERIODE_SONDAGE_S = 1.0


def sante_ok() -> bool:
    """L'API locale repond-elle sur /v1/health ? (jamais bloquant)"""
    try:
        with urllib.request.urlopen(URL_SANTE, timeout=1.5) as r:
            return r.status == 200
    except OSError:
        return False


def attendre_sante(delai: float = DELAI_SANTE_S) -> bool:
    """Attend que l'API reponde, au plus `delai` secondes."""
    fin = time.time() + delai
    while time.time() < fin:
        if sante_ok():
            return True
        time.sleep(0.5)
    return False


def demarrer_api() -> subprocess.Popen:
    """Demarre l'API dans un processus a part (meme port, meme environnement)."""
    env = {**os.environ, "AURA_API_PORT": str(PORT)}
    return subprocess.Popen([sys.executable, "-m", "aura.api"], env=env)


def surveiller(api: subprocess.Popen) -> None:
    """Boucle de supervision : relance l'API si elle s'arrete (backoff).

    Le nombre de relances est BORNE : un cerveau qui plante au chargement
    (GGUF absent, port pris) ne doit pas etre relance indefiniment — au-dela
    de MAX_RELANCES on abandonne en le disant.
    """
    relances = 0
    while True:
        if api.poll() is not None:
            if relances >= MAX_RELANCES:
                print(f"[superviseur] API arretee {relances} fois : abandon.",
                      file=sys.stderr)
                return
            relances += 1
            delai = min(2 ** relances, 30)
            print(f"[superviseur] API arretee (code {api.returncode}) : "
                  f"relance {relances}/{MAX_RELANCES} dans {delai}s",
                  file=sys.stderr)
            time.sleep(delai)
            api = demarrer_api()
            attendre_sante()
        time.sleep(PERIODE_SONDAGE_S)


def arreter(api: subprocess.Popen | None) -> None:
    """Arrete proprement l'API (terminate, puis kill si elle s'accroche)."""
    if api is None or api.poll() is not None:
        return
    api.terminate()
    try:
        api.wait(timeout=10)
    except subprocess.TimeoutExpired:
        api.kill()


def _attendre_interruption() -> None:
    """Reste en vie jusqu'a Ctrl+C.

    Cas d'une API lancee par un AUTRE processus : il n'y a rien a surveiller
    (on ne peut pas la relancer), mais l'appelant veut garder le cerveau en
    marche — comportement de l'ancien `main.py --api`.
    """
    while True:
        time.sleep(3600)


def lancer(interface: bool = True) -> int:
    """Amene l'API en vie (ou la REUTILISE), puis ouvre l'interface demandee.

    Renvoie le code de sortie du lanceur : 0 pour une fin normale, 1 quand
    l'API ne repond pas apres son demarrage (on n'ouvre rien dans ce cas).
    """
    api: subprocess.Popen | None = None
    if sante_ok():
        print(f"[superviseur] une API repond deja sur {URL_SANTE} : "
              "je la reutilise.")
    else:
        api = demarrer_api()
        if not attendre_sante():
            print("[superviseur] l'API ne repond pas : arret.", file=sys.stderr)
            arreter(api)
            return 1
        print(f"[superviseur] API prete sur 127.0.0.1:{PORT}")
    try:
        if not interface:
            if api is None:
                _attendre_interruption()
                return 0
            surveiller(api)
            return 0
        from . import tui
        return tui.lancer()
    except KeyboardInterrupt:
        print("\n[superviseur] arret demande.")
        return 0
    finally:
        arreter(api)
