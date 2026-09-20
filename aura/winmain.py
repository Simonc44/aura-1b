"""Aura.exe — point d'entree du paquet (winmain.py en exe).

Trois modes :
  1. sans argument  : lance la CLI python -m aura (mode demo/chat)
  2. <fichier.aef> [question] : boot COMPLET (verification + extraction +
     import du systeme) — utilisable directement si python+deps locales OK
  3. --cache [question] : le cache .cache_aef/ est deja rempli (par le
     boot du 2), import direct : demarrage instantane

En exe gele, kernel.py et chiffrement.py sont embarques en top-level
(--add-data) : import direct, hors du paquet aura (jamais en conflit avec
le paquet aura extrait du .aef).
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# En exe gele : les fichiers embarques (kernel.py, chiffrement.py) sont
# extraits dans _MEIPASS — il faut y pointer sys.path pour les importer.
if getattr(sys, "frozen", False):
    _meipass: str = getattr(sys, "_MEIPASS", "")
    if _meipass:
        sys.path.insert(0, _meipass)


def _import_kernel():
    """Import du kernel : paquet (python) ou top-level (exe gele)."""
    try:
        from aura import kernel
        return kernel
    except ImportError:
        import kernel
        return kernel


def _python_local():
    """Le python de l'environment uv local (dependances lourdes dispo)."""
    if not getattr(sys, "frozen", False):
        return sys.executable            # mode python : lui-meme
    projets = Path.home() / "Desktop" / "Dev" / "IA" / "aura-1b"
    candidats = [projets / ".venv" / "Scripts" / "python.exe",
                 shutil.which("python"), shutil.which("py")]
    for c in candidats:
        if c and Path(c).exists():
            return str(c)
    return None


def _executer_python(py: str, argv: list[str]) -> int:
    """Delegue au python local : verifie+extrait, puis execute la question.

    AURA_CACHE transmet l'emplacement du cache (l'exe et le python partagent
    le meme .cache_aef/ -> verification faite UNE seule fois).
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               AURA_CACHE=str(Path(sys.executable).resolve().parent
                              / ".cache_aef"))
    mode = "--cache" if argv and argv[0] == "--cache" else None
    cmd = [py, "-c", _SCRIPT_DELEGUE]
    if mode:
        cmd += ["--cache"]
        argv = argv[1:]
    cmd += argv
    return subprocess.call(cmd, env=env)


_SCRIPT_DELEGUE = r'''
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "aura") if False else os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
try:
    from aura import kernel
except ImportError:
    import kernel

args = sys.argv[1:]
if args and args[0] == "--cache":
    sysm = kernel.boot_depuis_cache()
    reste = args[1:]
elif args:
    sysm = kernel.boot(args[0])
    reste = args[1:]
else:
    sysm = None
    reste = []

if sysm is None:
    from aura.__main__ import main as cli_main
    sys.exit(cli_main([]))

ia = sysm["module"].Aura1B()
if reste:
    t0 = time.time()
    print(ia.executer(" ".join(reste)))
    print("[%0.1fs]" % (time.time() - t0))
else:
    print('Aura online. "exit" pour quitter, "reset" pour oublier.')
    while True:
        try:
            q = input("toi> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "reset":
            ia.reinitialiser_conversation()
            continue
        t0 = time.time()
        print(ia.executer(q))
        print("[%0.1fs]" % (time.time() - t0))
'''


def main() -> int:
    args = sys.argv[1:]
    mod_kernel = _import_kernel()

    # Exe gele : AURA_CACHE doit exister des maintenant (le python delegue
    # lira le meme cache) ; en python, kernel.CACHE suffit.
    if getattr(sys, "frozen", False):
        os.environ.setdefault(
            "AURA_CACHE",
            str(Path(sys.executable).resolve().parent / ".cache_aef"))

    # 1. pas d'argument -> CLI interactive complete (chat, demo, questions)
    if not args or args[0] in ("-h", "--help"):
        print("Aura-1B — usage :")
        print("  Aura.exe <fichier.aef> [question]   boot + reponse")
        print("  Aura.exe --cache [question]         cache deja rempli (instantane)")
        print("  Aura.exe                            interface interactive")
        if getattr(sys, "frozen", False):
            # le paquet aura n'est pas embarque dans l'exe leger -> CLI locale
            py = _python_local()
            if py:
                return _executer_python(py, [])
            return 0
        try:
            from aura.__main__ import main as cli_main
            return cli_main([])
        except Exception:
            return mod_kernel.principal()

    # Verification des entrees AVANT tout (echec rapide et lisible)
    if args[0] != "--cache":
        chemin = Path(args[0])
        if not chemin.exists():
            print(f"[erreur] fichier introuvable : {chemin}")
            return 2

    # 2. boot in-process si possible (python complet), sinon delegation
    # NB : import DYNAMIQUE (concatenation) -> PyInstaller ne detecte pas
    # llama_cpp statiquement et ne l'embarque pas (l'exe reste leger) ;
    # l'import reussit seulement si les deps lourdes sont installees.
    try:
        import importlib
        importlib.import_module("llama_" + "cpp")
        if args[0] == "--cache":
            sysm = mod_kernel.boot_depuis_cache()
            reste = args[1:]
        else:
            sysm = mod_kernel.boot(args[0])
            reste = args[1:]
        ia = sysm["module"].Aura1B()
    except ImportError:
        # exe leger : delegation au python local (uv) qui a les dependances
        py = _python_local()
        if not py:
            print("[erreur] python local introuvable pour l'execution "
                  "(l'exe ne verifie/extrait que le .aef)")
            return 3
        print(f"[exe] delegation au python local : {py}")
        return _executer_python(py, args)

    if reste:
        t0 = time.time()
        print(ia.executer(" ".join(reste)))
        print(f"[{time.time() - t0:.1f}s]")
        return 0

    print('Aura online. "exit" pour quitter, "reset" pour oublier.')
    while True:
        try:
            q = input("toi> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if q.lower() in ("exit", "quit"):
            return 0
        if q.lower() == "reset":
            ia.reinitialiser_conversation()
            continue
        t0 = time.time()
        print(ia.executer(q))
        print(f"[{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    sys.exit(main())
