"""Expert « PoT-code » : le 1B ecrit une fonction + des tests, la sandbox verifie.

Inspiré de Program-of-Thoughts et du self-debug : sur une demande de code,
le modele ecrit une fonction `resoudre` accompagnee d'asserts DANS LE MEME
BLOC. La sandbox execute tout : un assert qui echoue = code refuse — le
code rate ne sort JAMAIS, le pipeline retombe sur la generation normale.

Securite (honnete) :
- builtins restrictifs : pas d'import, pas d'open/eval/exec/compile/getattr,
  pas d'acces aux attributs doubles (obj.__class__) ;
- budget d'instructions par sys.settrace : toute boucle infinie est coupee
  (kill-switch garanti, pas un simple timeout de thread) ;
- extrait UNIQUEMENT du code entre balises ```python ... ``` ou ``` ... ```.
"""
import logging
import re
import sys

LOG = logging.getLogger("aura.potcode")

_BUDGET_INSTRUCTIONS = 200_000    # ≈ 2-5 s de pur Python, largement assez

_SYSTEME_CODE = (
    "Tu ecris du code Python. Reponds UNIQUEMENT avec un bloc de code :\n"
    "```python\n"
    "def resoudre(donnees):\n"
    "    \"\"\"Resout le probleme demande.\"\"\"\n"
    "    ...ton code...\n"
    "    return resultat\n"
    "\n"
    "# tests : au moins 3 asserts sur resoudre avec des exemples concrets\n"
    "assert resoudre(...) == ...\n"
    "```\n"
    "Regles :\n"
    "- la fonction s'appelle resoudre et prend les donnees du probleme\n"
    "- pas d'import, pas d'entrees/sorties, pas d'acces a des fichiers\n"
    "- les asserts doivent couvrir les cas du probleme, cas limites inclus\n"
    "- rien hors du bloc de code"
)

_BLOC_CODE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

# builtins autorises : arithmetique, structures de donnees, exceptions —
# tout ce dont un algorithme a besoin, rien de ce qui touche au systeme
_SANDBOX_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "float": float, "int": int,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "pow": pow, "range": range, "reversed": reversed, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "Exception": Exception, "ValueError": ValueError,
    "TypeError": TypeError, "IndexError": IndexError, "KeyError": KeyError,
    "ZeroDivisionError": ZeroDivisionError,
}


def extraire_code(texte: str) -> str | None:
    """Extrait le premier bloc de code balise, ou None."""
    m = _BLOC_CODE.search(texte)
    if not m:
        return None
    code = m.group(1).strip()
    return code or None


class _BudgetEpuise(Exception):
    pass


def _executer_sandbox(code: str,
                      budget: int = _BUDGET_INSTRUCTIONS) -> dict:
    """Compile et execute le code (definition + asserts) dans la sandbox.

    Renvoie l'environnement d'execution. Leve ValueError si le code est
    refuse : syntaxe, instruction illicite, budget depasse, assert rate.
    """
    try:
        arbre = compile(code, "<aura-sandbox>", "exec")
    except SyntaxError as e:
        raise ValueError(f"syntaxe invalide : {e}") from e

    env: dict = {"__builtins__": dict(_SANDBOX_BUILTINS)}
    compteur = {"n": 0}

    def _trace(frame, event, arg):
        if event == "line":
            compteur["n"] += 1
            if compteur["n"] > budget:
                raise _BudgetEpuise()
        return _trace

    sys.settrace(_trace)
    try:
        exec(arbre, env)            # noqa: S102 — sandbox construite pour
    except _BudgetEpuise:
        raise ValueError("budget d'instructions depasse (boucle infinie ?)")
    except AssertionError as e:
        raise ValueError(f"assert echoue : {e or '(sans message)'}")
    except Exception as e:
        raise ValueError(f"execution illicite : {type(e).__name__}: {e}")
    finally:
        sys.settrace(None)
    return env


def verifier_code(code: str, budget: int = _BUDGET_INSTRUCTIONS) -> dict:
    """Chaine complete : extraction deja faite -> sandbox + asserts.

    Un assert qui echoue leve ValueError : le code rate ne sort jamais.
    Renvoie un rapport avec la fonction validee.
    """
    env = _executer_sandbox(code, budget)
    fn = env.get("resoudre")
    if not callable(fn):
        raise ValueError("fonction resoudre absente")
    nb_asserts = sum(1 for l in code.splitlines()
                     if l.strip().startswith("assert "))
    if nb_asserts == 0:
        raise ValueError("aucun assert fourni")
    LOG.info("[potcode] code valide : %d asserts passes", nb_asserts)
    return {"fonction": fn, "nb_asserts": nb_asserts, "statut": "valide"}
