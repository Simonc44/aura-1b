"""Interface Aura basée sur pywebview : le rendu (fenêtre macOS, bulles de
chat, panneau Aperçu) est fait entièrement en HTML/CSS/JS (voir web/index.html,
repris de ta maquette). Ce fichier Python ne fait plus que :

  - charger le cerveau Aura1B en tâche de fond (comme avant),
  - exposer une petite API JS <-> Python via pywebview (classe Api),
  - piloter la fenêtre (déplacement, réduction, agrandissement, fermeture),
    puisque la fenêtre est "frameless" : la barre de titre custom est en HTML,
    donc plus de bouton système, il faut recréer ce comportement côté Python.

Installation : pip install pywebview
Usage        : python -m aura.gui_web   (--smoke : ferme seul apres 2,5 s)
               python aura/gui_web.py   (marche aussi, depuis n'importe quel
               dossier : la racine du projet est ajoutee a sys.path)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:      # uniquement pour les annotations (import relou en runtime)
    import webview

_ICI = Path(__file__).resolve().parent
_RACINE = _ICI.parent                      # racine du projet (contient aura/)
# Lancement en script direct (python aura/gui_web.py, double-clic...) :
# sys.path[0] est alors aura/, pas la racine -> import aura impossible.
if str(_RACINE) not in sys.path:
    sys.path.insert(0, str(_RACINE))

# NB : config/maj sont importes LOCALEMENT (dans main/_verifier_maj) :
# un import top-level de `aura.*` execute aura/__init__.py -> orchestrateur
# -> gplearn/llama_cpp, absents du Python global AVANT la relance venv.

_INDEX_HTML = _ICI / "web" / "index.html"

# Dependencies du projet : si l'interpreteur courant n'en a pas (double-clic
# sur le .py avec le Python global de Windows, qui n'a ni gplearn ni
# llama_cpp), on se relance automatiquement avec .venv, sinon message clair.
_DEPS = ("webview", "numpy", "sklearn", "gplearn", "llama_cpp")


def _python_venv() -> Path | None:
    for cand in (_RACINE / ".venv" / "Scripts" / "python.exe",
                 _RACINE / ".venv" / "bin" / "python"):
        if cand.is_file():
            return cand
    return None


def _manquantes() -> list[str]:
    import importlib.util
    return [m for m in _DEPS if importlib.util.find_spec(m) is None]


def _relancer_si_besoin() -> None:
    """Relance avec .venv si des modules du projet manquent (aucune boucle :
    dans la venv on n'est deja plus 'hors venv')."""
    manquantes = _manquantes()
    if not manquantes:
        return
    deja_venv = getattr(sys, "base_prefix", sys.prefix) != sys.prefix
    py = _python_venv()
    if py is not None and not deja_venv:
        import subprocess
        print(f"[gui_web] {', '.join(manquantes)} absent(s) de "
              f"{sys.executable} -> relance avec {py}")
        raise SystemExit(subprocess.call(
            [str(py), str(Path(__file__).resolve()), *sys.argv[1:]]))
    raise SystemExit(
        "[gui_web] modules manquants : " + ", ".join(manquantes) + "\n"
        "Installe-les avec :  uv sync\n"
        "Puis lance avec    :  .venv\\Scripts\\python.exe -m aura.gui_web")


class Api:
    """Pont exposé à JS sous `window.pywebview.api`.

    Chaque méthode publique (sans _ devant) est appelable depuis le JS via
    `pywebview.api.nom_methode(args)` et renvoie une Promise côté JS.
    pywebview exécute chaque appel dans son propre thread : un appel bloquant
    (le temps qu'Aura réfléchisse) ne fige donc pas la fenêtre.
    """

    def __init__(self) -> None:
        self._ia: Any = None          # Aura1B charge en tache de fond
        self._fenetre: "webview.Window | None" = None

    # ---------------- cycle de vie de la fenêtre (barre custom) ----------------
    def attacher_fenetre(self, fenetre: "webview.Window") -> None:
        self._fenetre = fenetre

    def fermer(self) -> None:
        if self._fenetre:
            self._fenetre.destroy()

    def reduire(self) -> None:
        if self._fenetre:
            self._fenetre.minimize()

    def agrandir(self) -> None:
        if self._fenetre:
            self._fenetre.toggle_fullscreen()

    def deplacer(self, dx: int, dy: int) -> None:
        """Déplace la fenêtre de (dx, dy) pixels — appelé en continu pendant
        le drag de la barre de titre HTML (voir web/index.html)."""
        if not self._fenetre:
            return
        try:
            self._fenetre.move(self._fenetre.x + int(dx), self._fenetre.y + int(dy))
        except Exception:  # noqa: BLE001
            pass

    # ---------------- chargement du cerveau ----------------
    def charger_ia(self) -> None:
        threading.Thread(target=self._charger_ia_thread, daemon=True).start()

    def _charger_ia_thread(self) -> None:
        t0 = time.time()
        try:
            from aura.orchestrateur import Aura1B

            ia = Aura1B()
            try:  # pré-charge la réponse de démarrage, cache froid absorbé
                ia.executer_detaille("Bonjour")
            except Exception:  # noqa: BLE001
                pass
            self._ia = ia
            self._pousser("aura_pret", {"latence": round(time.time() - t0, 1)})
            self._verifier_maj()
        except Exception as e:  # noqa: BLE001
            self._pousser("aura_erreur", {"message": str(e)})

    def _verifier_maj(self) -> None:
        """Pousse 'maj_disponible' vers le JS si une release plus recente
        existe (3 s max, silencieux hors-ligne — AURA_MAJ=0 pour couper)."""
        try:
            try:
                from . import maj
            except ImportError:           # lancement en script direct
                from aura import maj
            rap = maj.verifier(timeout=3.0)
            if rap.get("dispo"):
                self._pousser("maj_disponible",
                              {"version": rap.get("version"),
                               "nouvelle": rap.get("nouvelle"),
                               "url": rap.get("url")})
        except Exception:  # noqa: BLE001
            pass

    # ---------------- question / réponse ----------------
    def envoyer(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {"ok": False, "erreur": "question vide"}
        if self._ia is None:
            return {"ok": False, "erreur": "IA non prête"}
        t0 = time.time()
        try:
            r = self._ia.executer_detaille(question)
            return {
                "ok": True,
                "reponse": str(r.get("reponse", "")) or "...",
                "experts": sorted(r.get("experts") or set()),
                "cerveau": str(r.get("cerveau_choisi", "-")),
                "latence": round(time.time() - t0, 1),
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "erreur": str(e)}

    # ---------------- pousser un évènement vers JS depuis un thread Python ----------------
    def _pousser(self, evenement: str, donnees: dict) -> None:
        if not self._fenetre:
            return
        js = (
            "window.auraEvenement && window.auraEvenement("
            f"{json.dumps(evenement)}, {json.dumps(donnees)})"
        )
        try:
            self._fenetre.evaluate_js(js)
        except Exception:  # noqa: BLE001
            pass


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="aura.gui_web")
    ap.add_argument("--smoke", action="store_true", help="ferme seul apres 2,5 s (test)")
    args = ap.parse_args(argv)

    _relancer_si_besoin()          # bons modules ? sinon relance .venv
    # config APRES la verification des deps (voir commentaire module)
    try:
        from . import config as _config
    except ImportError:             # lancement en script direct
        from aura import config as _config
    _config.appliquer()            # aura.toml + profils, avant l'orchestrateur
    import webview  # pip install pywebview (apres verification des deps)

    api = Api()
    fenetre = webview.create_window(
        "Aura",
        url=str(_INDEX_HTML),
        js_api=api,
        width=980,
        height=680,
        frameless=True,
        easy_drag=False,  # le drag est géré en JS via api.deplacer (voir index.html)
        resizable=True,
        background_color="#E6E6E6",
    )
    assert fenetre is not None, "create_window a renvoye None"
    api.attacher_fenetre(fenetre)

    def au_demarrage() -> None:
        api.charger_ia()
        if args.smoke:
            def _fermer_apres() -> None:
                time.sleep(2.5)
                fenetre.destroy()

            threading.Thread(target=_fermer_apres, daemon=True).start()

    webview.start(au_demarrage, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
