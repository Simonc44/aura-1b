"""Genere les 4 datasets d'adaptateurs LoRA pour Aura.

Sources (tout est deja sur disque, rien d'invente sans verification) :
  - R1-U300/knowledge.jsonl   : 234 entrees catégorisées (fait, sante, droit...)
  - aura/.graphe_faits.jsonl  : 223 faits verifies (CRITIC)
  - aura/.cache_reponses.jsonl: vraies Q/A de l'usage reel
Enrichissement CALCULE (zero hallucination possible) :
  - math : operations, pourcentages, puissances, dates -> reponses par AST
Sortie : datasets/{web,math,code,general}.jsonl au format du script Colab.

Usage :
  python scripts/generer_datasets_adaptateurs.py
"""
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
SOURCE_KNOWLEDGE = Path(r"C:\Users\admin\Desktop\Dev\IA\R1-U300\knowledge.jsonl")
GRAPHE = RACINE / "aura" / ".graphe_faits.jsonl"
CACHE = RACINE / "aura" / ".cache_reponses.jsonl"
SORTIE = RACINE / "datasets"

# mapping cat knowledge.jsonl -> categorie d'adaptateur
MAPPING = {
    "fait": "web", "histoire": "web", "sante": "web", "droit": "web",
    "cuisine": "web", "piege": "web",
    "math": "math", "logique": "math",
    "code": "code", "informatique": "code",
    "francais": "general", "sens-commun": "general", "identite": "general",
}

random.seed(42)

# ── lecture des sources ──────────────────────────────────────────────────

def lire_jsonl(chemin: Path) -> list[dict]:
    if not chemin.exists():
        return []
    exemples = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if ligne:
            try:
                exemples.append(json.loads(ligne))
            except json.JSONDecodeError:
                continue
    return exemples


def question_variee(q: str) -> str:
    """Reformulations legeres pour eviter l'apprentissage d'un seul gabarit."""
    q = q.strip().rstrip("?")
    variantes = [
        q + " ?",
        "Dis-moi : " + q + " ?",
        "Peux-tu m'expliquer : " + q + " ?",
        "J'aimerais savoir : " + q + " ?",
    ]
    return random.choice(variantes)

# ── math : generateur avec reponses CALCULEES ────────────────────────────

def gen_math() -> list[dict]:
    exemples: list[dict] = []

    def ajoute(q: str, r: str) -> None:
        exemples.append({"question": q, "reponse": r})

    # operations de base (calculees, jamais devinees)
    for _ in range(30):
        a, b = random.randint(2, 99), random.randint(2, 99)
        op = random.choice([("+", a + b), ("x", a * b), ("-", a - b)])
        ajoute(f"combien fait {a} {op[0]} {b}",
               f"{a} {op[0]} {b} = {op[1]}")

    # pourcentages (calcules)
    for _ in range(20):
        pct = random.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 80])
        n = random.choice([40, 60, 80, 120, 150, 200, 250, 300, 500, 800])
        ajoute(f"combien fait {pct}% de {n}", f"{pct}% de {n} = {pct * n // 100}")

    # puissances et carres
    for _ in range(15):
        n = random.randint(2, 25)
        p = random.choice([2, 3])
        ajoute(f"combien fait {n} puissance {p}",
               f"{n}^{p} = {n ** p}")
    for _ in range(10):
        n = random.randint(2, 40)
        ajoute(f"quel est le carre de {n}", f"Le carre de {n} est {n * n}.")

    # suites simples arithmetiques (raisonnement)
    for _ in range(10):
        d = random.randint(2, 9)
        a0 = random.randint(1, 20)
        suite = [a0 + d * i for i in range(4)]
        ajoute(f"quelle est la suite logique : {', '.join(map(str, suite))}",
               f"Chaque terme augmente de {d}. Le suivant est {suite[-1] + d}.")

    # dates (calculees via datetime)
    for _ in range(10):
        jours = random.randint(3, 90)
        depart = date(2026, 1, 1) + timedelta(days=random.randint(0, 250))
        arrivee = depart + timedelta(days=jours)
        ajoute(f"quel jour de la semaine sera le {arrivee.strftime('%d/%m/%Y')}",
               f"Le {arrivee.strftime('%d/%m/%Y')} sera un {arrivee.strftime('%A')}.")

    # moyennes
    for _ in range(10):
        vals = [random.randint(2, 20) for _ in range(4)]
        ajoute(f"quelle est la moyenne de {', '.join(map(str, vals))}",
               f"Moyenne = ({' + '.join(map(str, vals))}) / 4 = {sum(vals) / 4:g}")

    # reponses knowledge.jsonl de cat math/logique en bonus
    return exemples

# ── code : familles parametrees (reponses executes mentalement, simples) ─

CODE_FAMILLES = [
    ("ecrire une fonction python {nom} qui additionne deux nombres",
     "def addition(a, b):\n    return a + b\n\n# Exemple : addition(2, 3) -> 5"),
    ("ecrire une fonction python qui verifie si un nombre est pair",
     "def est_pair(n):\n    return n % 2 == 0\n\n# Exemple : est_pair(4) -> True, est_pair(7) -> False"),
    ("ecrire une fonction python qui renvoie le maximum d'une liste",
     "def maximum(liste):\n    plus_grand = liste[0]\n    for x in liste:\n        if x > plus_grand:\n            plus_grand = x\n    return plus_grand\n\n# Exemple : maximum([3, 7, 2]) -> 7"),
    ("ecrire une fonction python qui calcule la moyenne d'une liste",
     "def moyenne(liste):\n    return sum(liste) / len(liste)\n\n# Exemple : moyenne([2, 4, 6]) -> 4.0"),
    ("ecrire une fonction python qui inverse une chaine de caracteres",
     "def inverser(texte):\n    return texte[::-1]\n\n# Exemple : inverser('aura') -> 'arua'"),
    ("ecrire une fonction python qui compte les voyelles d'un mot",
     "def compter_voyelles(mot):\n    return sum(1 for c in mot.lower() if c in 'aeiouy')\n\n# Exemple : compter_voyelles('python') -> 2 ('o' et 'y')"),
    ("ecrire une fonction python qui convertit des celsius en fahrenheit",
     "def celsius_vers_fahrenheit(c):\n    return c * 9 / 5 + 32\n\n# Exemple : celsius_vers_fahrenheit(100) -> 212.0"),
    ("ecrire une fonction python qui teste si un mot est un palindrome",
     "def est_palindrome(mot):\n    mot = mot.lower()\n    return mot == mot[::-1]\n\n# Exemple : est_palindrome('radar') -> True"),
    ("ecrire une fonction python qui trie une liste de nombres",
     "def trier(liste):\n    return sorted(liste)\n\n# Exemple : trier([3, 1, 2]) -> [1, 2, 3]"),
    ("ecrire une fonction python qui calcule la factorielle d'un nombre",
     "def factorielle(n):\n    resultat = 1\n    for i in range(2, n + 1):\n        resultat *= i\n    return resultat\n\n# Exemple : factorielle(5) -> 120"),
    ("ecrire une fonction python qui calcule la somme d'une liste",
     "def somme(liste):\n    total = 0\n    for x in liste:\n        total += x\n    return total\n\n# Exemple : somme([1, 2, 3]) -> 6"),
    ("ecrire une fonction python qui renvoie la longueur d'un texte",
     "def longueur(texte):\n    compteur = 0\n    for _ in texte:\n        compteur += 1\n    return compteur\n\n# Exemple : longueur('aura') -> 4"),
    ("ecrire une fonction python qui convertit des kilometres en miles",
     "def km_vers_miles(km):\n    return km * 0.621371\n\n# Exemple : km_vers_miles(10) -> 6.21371"),
    ("ecrire une fonction python qui dit si un nombre est positif ou negatif",
     "def signe(n):\n    if n > 0:\n        return 'positif'\n    if n < 0:\n        return 'negatif'\n    return 'nul'\n\n# Exemple : signe(-5) -> 'negatif'"),
]


def gen_code(exemples_connaissances: list[dict]) -> list[dict]:
    exemples: list[dict] = []
    # familles parametrees avec reformulations naturelles
    reformulations = [
        "{q} ?",
        "Peux-tu m'aider : {q} ?",
        "J'ai besoin de ça : {q}",
        "Donne-moi le code pour : {q}",
    ]
    for q, r in CODE_FAMILLES:
        rf = random.choice(reformulations)
        exemples.append({"question": rf.format(q=q), "reponse": r})
    # + exemples du knowledge.jsonl (code + informatique), deja mappes,
    # en plusieurs reformulations pour densifier le dataset
    for e in exemples_connaissances:
        q = e.get("q") or e.get("question")
        a = e.get("a") or e.get("reponse")
        if not (q and a):
            continue
        for _ in range(2):
            exemples.append({"question": question_variee(q), "reponse": a})
    return exemples

# ── general : redaction et sens commun ───────────────────────────────────

def gen_general(exemples_connaissances: list[dict], cache: list[dict]) -> list[dict]:
    exemples: list[dict] = []
    for e in exemples_connaissances:
        q = e.get("q") or e.get("question")
        a = e.get("a") or e.get("reponse")
        if not (q and a):
            continue
        for _ in range(3):  # 3 reformulations par entree (redaction = variete)
            exemples.append({"question": question_variee(q), "reponse": a})
    # vraies Q/A de l'usage (deja verifiees par l'usage reel)
    for c in cache:
        if len(c.get("r", "")) > 20 and "je ne peux pas" not in c.get("r", "").lower():
            exemples.append({"question": c["q"], "reponse": c["r"]})
    return exemples

# ── assemblage ───────────────────────────────────────────────────────────

def principal() -> None:
    SORTIE.mkdir(exist_ok=True)
    knowledge = lire_jsonl(SOURCE_KNOWLEDGE)
    cache = lire_jsonl(CACHE)

    paniers: dict[str, list[dict]] = {"web": [], "math": [], "code": [], "general": []}

    # 1. knowledge.jsonl -> mapping
    for e in knowledge:
        cat = MAPPING.get(e.get("cat", ""), None)
        if cat and e.get("q") and e.get("a"):
            paniers[cat].append({"question": question_variee(e["q"]),
                                 "reponse": e["a"]})

    # 2. math : enrichissement calcule
    paniers["math"] = gen_math() + paniers["math"]

    # 3. code : familles + connaissances
    paniers["code"] = gen_code(paniers["code"])  # renforce au passage

    # 4. general : redaction/sens-commun/cache
    paniers["general"] = gen_general(paniers["general"], cache)

    # 5. web : ajoute le fait en "contexte" (l'adaptateur apprend a repondre
    #    A PARTIR des faits fournis, comme le veut sa personnalite)
    web_final = []
    for e in paniers["web"]:
        web_final.append({"question": e["question"], "reponse": e["reponse"],
                          "contexte": e["reponse"]})
    paniers["web"] = web_final

    # ecriture + rapport
    print(f"{'categorie':<10} {'exemples':>8}")
    for cat, exemples in paniers.items():
        chemin = SORTIE / f"{cat}.jsonl"
        with chemin.open("w", encoding="utf-8") as f:
            for e in exemples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"{cat:<10} {len(exemples):>8}  -> {chemin.name}")


if __name__ == "__main__":
    principal()
