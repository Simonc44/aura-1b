"""Self-play : RontoGPT s'affronte elle-meme et distille ses reussites.

L'idee (n°4 des quatre poussees par l'utilisateur)
--------------------------------------------------
Deux instances du MEME cerveau, avec deux consignes opposees :

* le PIEgeur invente un piege arithmetique (enonce court, calcul exact) ;
* le SOLVEUR, lui, doit le resoudre en pensant dans sa balise de pensee.

Ce qui est garde
----------------
Seules les paires VERIFIEES deviennent des exemples d'entrainement :

1. le calcul du piegeur est reevalue EXACTEMENT (rationnels, cf.
   `_resoudre`) et doit egaler la reponse qu'il a lui-meme annoncee — un
   enonce dont on ne sait pas calculer la cible est JETE ;
2. le solveur doit trouver ce nombre ;
3. la pensee du solveur est conservee telle quelle (elle sert de cible
   `pensee` au re-entrainement LoRA, cf. scripts/colab_lora_gemma.ipynb).

Ce qui n'est PAS pretendu
-------------------------
Ce script n'est pas un juge de la verite. Il ne garde que des enonces dont
l'arithmetique est exacte ET coherente avec elle-meme ; il ne peut pas
garantir que l'ENONCE redige decrit bien le CALCUL annonce. C'est
precisement pourquoi le filtre est « reponse == evaluation exacte du calcul »
et non « le modele a l'air convaincu ». Un item invérifiable est jeté, pas
deviné.

Sortie : JSONL directement consommable par l'entraineur d'adaptateurs :
  {"question": "...", "reponse": "...", "pensee": "..."}

Usage :
  python scripts/selfplay_distille.py --n 40
  python scripts/selfplay_distille.py --n 40 --sortie datasets/selfplay.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import time
from fractions import Fraction
from pathlib import Path

# ── CONSIGNES DES DEUX ROLES ────────────────────────────────────────────

#: Le piegeur doit produire un format STRICT : sans ca, rien n'est
#: verifiable et l'item est jete. Trois lignes, rien d'autre.
CONSIGNE_PIEGEUR = (
    "Tu es un professeur qui fabrique des pieges de calcul. Invente UN "
    "probleme court en francais dont la reponse est un nombre, et dont le "
    "calcul est facile a rater (ordre des operations, pourcentage, "
    "distracteur). Reponds EXACTEMENT en trois lignes, sans rien d'autre :\n"
    "ENONCE: <le probleme>\n"
    "CALCUL: <l'expression arithmetique exacte de la bonne reponse, "
    "uniquement des nombres et + - * / ( )>\n"
    "REPONSE: <le nombre>\n"
)

#: Le solveur pense dans sa balise native, puis conclut par un nombre NU
#: sur la derniere ligne du tour `model`.
CONSIGNE_SOLVEUR = (
    "Tu es un solveur. Resous le probleme en raisonnant dans ta balise de "
    "pensee, puis termine ta reponse par une derniere ligne ne contenant que "
    "le nombre final, sans unite ni texte."
)


# ── EVALUATION EXACTE D'UNE EXPRESSION ARITHMETIQUE ─────────────────────
# Fraction (jamais float) : une erreur d'arrondi ne doit pas invalider un
# item valide, ni valider un item faux.
_BINAIRES = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}
_UNAIRES = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def _noeud(noeud: ast.AST, source: str) -> Fraction:
    """Valeur exacte d'un noeud AST restreint (nombres, + - * / **)."""
    if isinstance(noeud, ast.Expression):
        return _noeud(noeud.body, source)
    if isinstance(noeud, ast.Constant):
        texte = ast.get_source_segment(source, noeud)
        if texte is None or not re.fullmatch(r"\d+(?:\.\d+)?", texte.strip()):
            raise ValueError(f"nombre illisible : {texte!r}")
        return Fraction(texte.strip())
    if isinstance(noeud, ast.BinOp) and type(noeud.op) in _BINAIRES:
        gauche = _noeud(noeud.left, source)
        droite = _noeud(noeud.right, source)
        if isinstance(noeud.op, ast.Pow):
            # puissance entiere bornee : 2**100000 n'est pas un controle,
            # c'est un deni de service sur la memoire.
            if droite.denominator != 1 or not (0 <= droite <= 4096):
                raise ValueError("puissance non entiere ou hors bornes")
        return _BINAIRES[type(noeud.op)](gauche, droite)
    if isinstance(noeud, ast.UnaryOp) and type(noeud.op) in _UNAIRES:
        return _UNAIRES[type(noeud.op)](_noeud(noeud.operand, source))
    raise ValueError(f"noeud refuse : {type(noeud).__name__}")


def _resoudre(expression: str) -> Fraction | None:
    """Valeur EXACTE d'une expression, ou None si elle n'est pas calculable.

    None plutot qu'une exception : l'appelant lit du texte produit par un
    modele, il ne doit jamais planter dessus.
    """
    if not expression:
        return None
    expr = expression.strip().rstrip("=.").strip()
    if not expr or len(expr) > 200:
        return None
    try:
        arbre = ast.parse(expr, mode="eval")
        valeur = _noeud(arbre, expr)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError,
            OverflowError, MemoryError):
        return None
    return valeur


def _joli(valeur: Fraction) -> str:
    """Fraction -> texte lisible (54 plutot que 54/1)."""
    if valeur.denominator == 1:
        return str(valeur.numerator)
    return f"{float(valeur):.6g}"


# ── LECTURE DES REPONSES DU MODELE ──────────────────────────────────────

_RE_LIGNE = re.compile(r"^\s*(ENONCE|CALCUL|REPONSE)\s*:\s*(.+?)\s*$",
                       re.IGNORECASE | re.MULTILINE)
_RE_NOMBRE = re.compile(r"-?\d+(?:[.,]\d+)?")
#: balises de pensee natives du cerveau (assemblees : le litteral ne doit
#: pas etre confondu avec du balisage au moment de l'edition)
_OUV = "<" + "think>"
_FER = "<" + "/think>"


def _champs(texte: str) -> dict[str, str]:
    """Extrait les trois champs du piegeur (le dernier gagne si repete)."""
    champs: dict[str, str] = {}
    for m in _RE_LIGNE.finditer(texte or ""):
        champs[m.group(1).upper()] = m.group(2).strip()
    return champs


def _nombre_final(texte: str) -> Fraction | None:
    """Le DERNIER nombre ecrit (les modele concluent par la reponse)."""
    if not texte:
        return None
    nombres = _RE_NOMBRE.findall(texte)
    if not nombres:
        return None
    try:
        return Fraction(nombres[-1].replace(",", "."))
    except (ValueError, ZeroDivisionError):
        return None


def _equivalent(a: Fraction | None, b: Fraction | None) -> bool:
    return a is not None and b is not None and a == b


# ── BOUCLE DE SELF-PLAY ─────────────────────────────────────────────────

def _un_tour(question: str, systeme: str, think: bool,
             secondes: float) -> tuple[str, str]:
    """(reponse, pensee) du cerveau local, avec echeance dure."""
    from aura import llama_cerveau

    debut = time.time()
    reponse = llama_cerveau.generer(question, systeme=systeme, think=think,
                                    deadline=debut + secondes)
    pensee = llama_cerveau.reflexion_du_tour() or ""
    return reponse, pensee


def distiller(n: int = 20, sortie: str = "datasets/selfplay.jsonl",
              secondes: float = 60.0, graine: int | None = None) -> dict:
    """Joue n manches et n'ecrit QUE les reussites verifiees.

    Renvoie un compte-rendu honnete : combien de manches, combien de pieges
    exploitables, combien resolus, combien ecrits.
    """
    from aura import llama_cerveau

    if not llama_cerveau.disponible():
        return {"erreur": "cerveau local indisponible (modele absent ?)"}
    if graine is not None:
        random.seed(graine)

    themes = ["un pourcentage", "un ordre d'operations", "un calcul de "
              "vitesse", "une remise de prix", "un partage en parts egales",
              "une conversion d'unites", "un calcul de moyenne"]
    ecrites, exploitables, resolues = [], 0, 0
    chemin = Path(sortie)

    for manche in range(1, n + 1):
        theme = themes[manche % len(themes)]
        # 1) le PIEgeur
        brut, _ = _un_tour(f"Invente un piege sur {theme}.", CONSIGNE_PIEGEUR,
                           think=False, secondes=secondes)
        champs = _champs(brut)
        enonce, calcul = champs.get("ENONCE"), champs.get("CALCUL")
        cible = _resoudre(calcul) if calcul else None
        annonce = _nombre_final(champs.get("REPONSE", ""))
        if not enonce or not _equivalent(cible, annonce):
            print(f"[{manche}] piege jete (calcul non verifiable)")
            continue
        exploitables += 1

        # 2) le SOLVEUR, sur un enonce NU (sans le calcul, sinon il suffirait
        #    de recopier) et avec sa pensee native.
        question = re.sub(r"^\s*ENONCE\s*:\s*", "", enonce,
                          flags=re.IGNORECASE).strip()
        reponse, pensee = _un_tour(question, CONSIGNE_SOLVEUR, think=True,
                                   secondes=secondes)
        trouve = _nombre_final(reponse)
        if not _equivalent(trouve, cible):
            print(f"[{manche}] rate : attendu {_joli(cible)}, "
                  f"trouve {_joli(trouve) if trouve else '?'}")
            continue
        resolues += 1

        # 3) la pensee est GARDEE : c'est la cible `pensee` du LoRA.
        ecrites.append({"question": question, "reponse": _joli(cible),
                        "pensee": pensee.strip()})
        print(f"[{manche}] reussi : {_joli(cible)}")

    if ecrites:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if os.environ.get("AURA_SELFPLAY_APPEND") else "w"
        with open(chemin, mode, encoding="utf-8") as f:
            for e in ecrites:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return {"manches": n, "pieges_exploitables": exploitables,
            "resolus": resolues, "ecrits": len(ecrites),
            "fichier": str(chemin)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="nombre de manches")
    p.add_argument("--sortie", default="datasets/selfplay.jsonl")
    p.add_argument("--secondes", type=float, default=60.0,
                   help="echeance dure par generation")
    p.add_argument("--graine", type=int, default=None)
    a = p.parse_args()
    print(json.dumps(distiller(a.n, a.sortie, a.secondes, a.graine),
                     ensure_ascii=False, indent=2))
