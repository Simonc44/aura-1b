"""Construit aura/lang.db : le dictionnaire symbolique embarqué.

Sources (toutes déterministes, hors-ligne) :
  - verbes irréguliers (être, avoir, aller, faire, dire, pouvoir,
    vouloir, savoir, venir, voir) aux 4 temps
  - verbes en -er réguliers (radical + terminaisons exactes)
  - définitions exactes (mots courants, utiles au 1B)
  - exceptions d'élision (h aspiré : « le héros », pas « l'héros »)

Usage : uv run python scripts/construire_lang.py
La base est reconstruite à chaque évolution, puis voyage dans le .aef.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
BDD = RACINE / "aura" / "lang.db"

PERSONNES = ("je", "tu", "il", "nous", "vous", "ils")

# ── verbes irréguliers (formes SANS pronom : « serai », pas « je serai ») ──
IRREGULIERS: dict[str, dict[str, list[str]]] = {
    "être": {
        "present": ["suis", "es", "est", "sommes", "êtes", "sont"],
        "imparfait": ["étais", "étais", "était", "étions", "étiez", "étaient"],
        "futur": ["serai", "seras", "sera", "serons", "serez", "seront"],
        "passe_compose": ["ai été", "as été", "a été", "avons été",
                          "avez été", "ont été"],
    },
    "avoir": {
        "present": ["ai", "as", "a", "avons", "avez", "ont"],
        "imparfait": ["avais", "avais", "avait", "avions", "aviez", "avaient"],
        "futur": ["aurai", "auras", "aura", "aurons", "aurez", "auront"],
        "passe_compose": ["eu", "as eu", "a eu", "avons eu", "avez eu",
                          "ont eu"],
    },
    "aller": {
        "present": ["vais", "vas", "va", "allons", "allez", "vont"],
        "imparfait": ["allais", "allais", "allait", "allions", "alliez",
                      "allaient"],
        "futur": ["irai", "iras", "ira", "irons", "irez", "iront"],
        "passe_compose": ["suis allé", "es allé", "est allé", "sommes allés",
                          "êtes allés", "sont allés"],
    },
    "faire": {
        "present": ["fais", "fais", "fait", "faisons", "faites", "font"],
        "imparfait": ["faisais", "faisais", "faisait", "faisions", "faisiez",
                      "faisaient"],
        "futur": ["ferai", "feras", "fera", "ferons", "ferez", "feront"],
        "passe_compose": ["ai fait", "as fait", "a fait", "avons fait",
                          "avez fait", "ont fait"],
    },
    "dire": {
        "present": ["dis", "dis", "dit", "disons", "dites", "disent"],
        "imparfait": ["disais", "disais", "disait", "disions", "disiez",
                      "disaient"],
        "futur": ["dirai", "diras", "dira", "dirons", "direz", "diront"],
        "passe_compose": ["ai dit", "as dit", "a dit", "avons dit",
                          "avez dit", "ont dit"],
    },
    "pouvoir": {
        "present": ["peux", "peux", "peut", "pouvons", "pouvez", "peuvent"],
        "imparfait": ["pouvais", "pouvais", "pouvait", "pouvions", "pouviez",
                      "pouvaient"],
        "futur": ["pourrai", "pourras", "pourra", "pourrons", "pourrez",
                  "pourront"],
        "passe_compose": ["ai pu", "as pu", "a pu", "avons pu", "avez pu",
                          "ont pu"],
    },
    "vouloir": {
        "present": ["veux", "veux", "veut", "voulons", "voulez", "veulent"],
        "imparfait": ["voulais", "voulais", "voulait", "voulions", "vouliez",
                      "voulaient"],
        "futur": ["voudrai", "voudras", "voudra", "voudrons", "voudrez",
                  "voudront"],
        "passe_compose": ["ai voulu", "as voulu", "a voulu", "avons voulu",
                          "avez voulu", "ont voulu"],
    },
    "savoir": {
        "present": ["sais", "sais", "sait", "savons", "savez", "savent"],
        "imparfait": ["savais", "savais", "savait", "savions", "saviez",
                      "savaient"],
        "futur": ["saurai", "sauras", "saura", "saurons", "saurez", "sauront"],
        "passe_compose": ["ai su", "as su", "a su", "avons su", "avez su",
                          "ont su"],
    },
    "venir": {
        "present": ["viens", "viens", "vient", "venons", "venez", "viennent"],
        "imparfait": ["venais", "venais", "venait", "venions", "veniez",
                      "venaient"],
        "futur": ["viendrai", "viendras", "viendra", "viendrons", "viendrez",
                  "viendront"],
        "passe_compose": ["suis venu", "es venu", "est venu", "sommes venus",
                          "êtes venus", "sont venus"],
    },
    "voir": {
        "present": ["vois", "vois", "voit", "voyons", "voyez", "voient"],
        "imparfait": ["voyais", "voyais", "voyait", "voyions", "voyiez",
                      "voyaient"],
        "futur": ["verrai", "verras", "verra", "verrons", "verrez", "verront"],
        "passe_compose": ["ai vu", "as vu", "a vu", "avons vu", "avez vu",
                          "ont vu"],
    },
}

# ── verbes en -er : radical + terminaisons exactes ─────────────────────
ER = {
    "present": ["e", "es", "e", "ons", "ez", "ent"],
    "imparfait": ["ais", "ais", "ait", "ions", "iez", "aient"],
    "futur": ["erai", "eras", "era", "erons", "erez", "eront"],
    # passé composé avec avoir : participe passé (accord simplifié, forme
    # canonique « j'ai parlé »)
    "passe_compose": ["é", "as é", "a é", "avons é", "avez é", "ont é"],
}
VERBES_ER = ("manger", "parler", "donner", "aimer", "chercher", "travailler",
             "jouer", "écouter", "demander", "penser")

# ── définitions (exactes, concises) ────────────────────────────────────
DEFINITIONS: dict[str, str] = {
    "algorithme": "suite finie d'étapes précises pour résoudre un problème.",
    "ordinateur": "machine qui exécute automatiquement des programmes.",
    "molécule": "assemblage d'atomes liés entre eux.",
    "photosynthèse": "processus par lequel les plantes fabriquent leur "
                     "matière avec la lumière, l'eau et le CO2.",
    "gravitation": "force qui attire les masses entre elles.",
    "électricité": "déplacement de charges électriques dans un conducteur.",
    "démocratie": "régime où le pouvoir appartient au peuple qui l'exerce "
                  "par l'élection.",
    "métaphore": "figure de style qui associe un mot à un autre sans "
                 "outil de comparaison.",
    "adverbe": "mot invariable qui modifie un verbe, un adjectif ou un "
               "autre adverbe.",
    "conjugaison": "ensemble des formes que prend un verbe selon la "
                   "personne, le temps et le mode.",
    "syllabe": "unité de son prononcée d'un seul trait dans un mot.",
    "nom": "mot qui désigne une personne, un animal, une chose ou une idée.",
    "verbe": "mot qui exprime une action ou un état et se conjugue.",
    "adjectif": "mot qui précise une qualité du nom auquel il se rapporte.",
    "pronom": "mot qui remplace un nom (« il », « celle-ci »...).",
}

# ── exceptions d'élision (h aspiré : PAS d'élision) ────────────────────
H_ASPIRE = ("héros", "hache", "hameçon", "hamac", "haricot", "hareng",
            "hall", "hameau", "hanneton", "happer", "harpe", "hasard",
            "honteux", "huit", "huître", "hurler", "hussard")


def construire() -> int:
    if BDD.exists():
        BDD.unlink()
    conn = sqlite3.connect(BDD)
    conn.executescript("""
    CREATE TABLE conjugaisons (
        verbe TEXT NOT NULL, temps TEXT NOT NULL, personne TEXT NOT NULL,
        forme TEXT NOT NULL, PRIMARY KEY (verbe, temps, personne));
    CREATE TABLE dictionnaire (
        mot TEXT PRIMARY KEY, definition TEXT NOT NULL);
    CREATE TABLE elisions (mot TEXT PRIMARY KEY, elision TEXT NOT NULL);
    """)
    donnees = []
    for verbe, temps_map in IRREGULIERS.items():
        for temps, formes in temps_map.items():
            for pers, forme in zip(PERSONNES, formes):
                donnees.append((verbe, temps, pers, forme))
    for verbe in VERBES_ER:
        radical = verbe[:-2]
        for temps, terminaisons in ER.items():
            for pers, term in zip(PERSONNES, terminaisons):
                donnees.append((verbe, temps, pers, radical + term))
    conn.executemany("INSERT INTO conjugaisons VALUES (?,?,?,?)", donnees)
    conn.executemany("INSERT INTO dictionnaire VALUES (?,?)",
                     list(DEFINITIONS.items()))
    conn.executemany("INSERT INTO elisions VALUES (?,?)",
                     [(mot, "le") for mot in H_ASPIRE])
    conn.commit()
    n_verbes = conn.execute("SELECT COUNT(DISTINCT verbe) "
                            "FROM conjugaisons").fetchone()[0]
    n_defs = conn.execute("SELECT COUNT(*) FROM dictionnaire").fetchone()[0]
    n_el = conn.execute("SELECT COUNT(*) FROM elisions").fetchone()[0]
    conn.close()
    print(f"[lang] {BDD.name} : {n_verbes} verbes x 4 temps, "
          f"{n_defs} définitions, {n_el} exceptions d'élision "
          f"({BDD.stat().st_size / 1024:.0f} Ko)")
    return 0


if __name__ == "__main__":
    sys.exit(construire())
