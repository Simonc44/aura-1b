"""Point d'entree unique de RontoGPT.

    python main.py            # API surveillee + interface terminal (TUI)
    python main.py --api      # API seule, surveillee (relancee si elle plante)

Le lanceur ne contient aucune logique metier : il demarre l'API (`aura.api`)
comme processus independant, verifie sa sante sur /v1/health, la relance
avec un delai croissant si elle plante, puis ouvre le TUI qui se connecte a
cette meme API. A la fermeture, l'API est arretee proprement.
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request

PORT = int(os.environ.get("AURA_API_PORT", "8760"))
URL = f"http://127.0.0.1:{PORT}/v1/health"
MAX_RELANCES = 5


def sante_ok() -> bool:
    try:
        with urllib.request.urlopen(URL, timeout=1.5) as r:
            return r.status == 200
    except OSError:
        return False


def attendre_sante(delai: float = 60.0) -> bool:
    fin = time.time() + delai
    while time.time() < fin:
        if sante_ok():
            return True
        time.sleep(0.5)
    return False


def demarrer_api() -> subprocess.Popen:
    env = {**os.environ, "AURA_API_PORT": str(PORT)}
    return subprocess.Popen([sys.executable, "-m", "aura.api"], env=env)


def surveiller(api: subprocess.Popen) -> None:
    """Boucle de supervision : relance l'API si elle s'arrete (backoff)."""
    relances = 0
    while True:
        if api.poll() is not None:
            if relances >= MAX_RELANCES:
                print(f"[main] API arretee {relances} fois : abandon.")
                return
            relances += 1
            delai = min(2 ** relances, 30)
            print(f"[main] API arretee (code {api.returncode}) : "
                  f"relance {relances}/{MAX_RELANCES} dans {delai}s")
            time.sleep(delai)
            api = demarrer_api()
            attendre_sante()
        time.sleep(1)


def arreter(api: subprocess.Popen) -> None:
    if api.poll() is None:
        api.terminate()
        try:
            api.wait(timeout=10)
        except subprocess.TimeoutExpired:
            api.kill()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="main.py")
    p.add_argument("--api", action="store_true", help="API seule, sans TUI")
    args = p.parse_args(argv)

    if sante_ok():
        print(f"[main] une API repond deja sur {URL} : je la reutilise.")
        api = None
    else:
        api = demarrer_api()
        if not attendre_sante():
            print("[main] l'API ne repond pas : arret.", file=sys.stderr)
            arreter(api)
            return 1
        print(f"[main] API prete sur 127.0.0.1:{PORT}")

    if args.api:
        try:
            if api is not None:
                surveiller(api)
            else:
                while True:
                    time.sleep(3600)
        except KeyboardInterrupt:
            print("\n[main] arret demande.")
        finally:
            if api is not None:
                arreter(api)
        return 0

    try:
        from aura import tui
        return tui.lancer([])
    finally:
        if api is not None:
            arreter(api)


if __name__ == "__main__":
    sys.exit(main())
