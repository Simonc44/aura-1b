"""Banc de non-regression QUALITE d'Aura — 14 questions, un score global.

Pourquoi : on ameliorait le projet « a l'oeil ». Chaque changement de
prompt, de modele, de budget ou de routage se verifie ici en 1 commande,
avec un SCORE comparable d'un run a l'autre (regression visible tout de
suite).

Categories et ce qu'on y mesure :
    meta      -> doit repondre vite ET sans lancer la machine de guerre
    identite  -> constante (0 generation)
    exact     -> maths / calcul verbal : la BONNE valeur, sinon 0
    logique   -> enigme a solution unique : la solution attendue
    riche     -> dissertation : SANS boucle, SANS balise ecolee, SANS
                 refus en tete, SANS phrase copiee (cf. bug 153 s)
    factuel   -> pas de generation en boucle, garde-fou hors-ligne honnete

Le cache semantique du banc est RECRE A CHAQUE RUN (sauf --garder_cache) :
un hit servirait une vieille generation et masquerait une regression.

Usage :
    .venv/Scripts/python.exe scripts/bench_qualite.py              # tout
    .venv/Scripts/python.exe scripts/bench_qualite.py --rapide     # sans le riche (2 min)
    .venv/Scripts/python.exe scripts/bench_qualite.py --qid riche-puissance
    .venv/Scripts/python.exe scripts/bench_qualite.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

_RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RACINE))

# ISOLEMENT : le banc ne doit JAMAIS remplir le cache de reponses reel
# (un hit y ferait echouer les tests de routage — regression vue) ni la
# memoire que l'utilisateur partage avec l'application.
_CACHE_BANC = _RACINE / "bench_qualite_cache.jsonl"
os.environ["AURA_FILTRE_CACHE"] = str(_CACHE_BANC)

_REFUS_TETE = re.compile(
    r"^\s*(je ne peux pas|je ne suis pas en mesure|desole|pardon, mais)", re.I)


def _sans_boucle(reponse: str) -> bool:
    from aura import llama_cerveau as lc
    return not lc._en_boucle(reponse)


def _sans_balise(reponse: str) -> bool:
    return not re.search(r"SECTION A REDIGER|PARTIE [123] :|CONNECTEURS :",
                         reponse)


def _sans_copie_interne(reponse: str) -> bool:
    from aura import llama_cerveau as lc
    phrases = [p.lower() for p in lc._phrases(reponse) if len(p) >= 50]
    return len(phrases) == len(set(phrases))


def _contient(reponse: str, *morceaux: str) -> bool:
    bas = re.sub(r"\s+", " ", reponse.lower())
    return all(m.lower() in bas for m in morceaux)


def _defauts_riche(reponse: str) -> list[str]:
    """Liste des defauts d'une dissertation (liste vide = reponse saine).

    Renvoie le NOM de chaque controle en echec — un « KO » seul ne dit pas
    ce qu'il faut corriger (vu au banc : balise ecolee vs phrase copiee).
    """
    defauts = []
    if not _sans_boucle(reponse):
        defauts.append("boucle")
    if not _sans_balise(reponse):
        defauts.append("balise")
    if not _sans_copie_interne(reponse):
        defauts.append("copie")
    if _REFUS_TETE.search(reponse):
        defauts.append("refus")
    if len(reponse) <= 300:
        defauts.append("court")
    return defauts


def _controle_riche(reponse: str, meta: dict) -> tuple[bool, str]:
    defauts = _defauts_riche(reponse)
    mesure = f"{meta['latence']:.0f}s, {len(reponse)} car"
    if defauts:
        mesure += "  KO:" + ",".join(defauts)
    return (not defauts, mesure)


# (id, categorie, question, controle) — controle(reponse, meta) -> (ok, detail)
QUESTIONS: list[tuple[str, str, str, object]] = [
    # ── meta : le bug historique (153 s de dissertation hors-sujet) ──
    ("meta-8b", "meta", "es tu une ia aussi puissn qu'un 8B ou pluas",
     lambda r, m: (m["latence"] < 5.0 and _sans_balise(r) and len(r) < 900,
                   f"{m['latence']:.2f}s, {len(r)} car")),
    ("meta-capacite", "meta", "Tu es plus fort que ChatGPT ?",
     lambda r, m: (m["latence"] < 5.0 and _sans_balise(r), f"{m['latence']:.2f}s")),
    ("meta-long", "meta",
     "Est-ce que tu es une intelligence artificielle vraiment capable de "
     "comprendre ce que tu ecris, ou tu simules seulement ?",
     lambda r, m: (m["latence"] < 40 and _sans_balise(r) and _sans_boucle(r),
                   f"{m['latence']:.1f}s")),

    # ── identite : constante, zero generation ──
    ("identite-nom", "identite", "Qui es-tu ?",
     lambda r, m: (m["cerveau"] == "constante" and "Aura" in r,
                   m["cerveau"])),
    ("identite-createur", "identite", "Qui t'a cree ?",
     lambda r, m: (m["cerveau"] == "constante" and "Simon" in r, m["cerveau"])),

    # ── exact : la bonne valeur ou 0 ──
    ("math-carre", "exact", "Combien font 6 x 7 ?",
     lambda r, m: ("42" in r, r[:40])),
    ("math-puissance", "exact", "Quel est 2 puissance 10 ?",
     lambda r, m: ("1024" in re.sub(r"\s", "", r), r[:40])),
    ("verbal-pommes", "exact",
     "J'ai 3 pommes, j'en mange 1, combien il m'en reste ?",
     lambda r, m: ("2" in r, r[:40])),
    ("verbal-trois", "exact", "Si j'ai 10 euros et que je dépense 4 euros, "
     "combien me reste-t-il ?",
     lambda r, m: ("6" in r, r[:40])),

    # ── logique : solution unique ──
    ("logique-ages", "logique",
     "Alice a 3 ans de plus que Bob. Bob a 2 ans de plus que Clara. "
     "Clara a 4 ans. Quel age a Alice ?",
     lambda r, m: ("9" in re.sub(r"[^\d]", "", r) or "neuf" in r.lower(),
                   r[:60])),

    # ── riche : la dissertation, sans les defauts observes en prod ──
    ("riche-solaire", "riche",
     "Redige une analyse comparee du solaire et de l'eolien en France, "
     "avec les avantages et les limites de chaque filiere",
     _controle_riche),
    ("riche-puissance", "riche",
     "La puissance de la technologie est-elle une menace pour l'homme "
     "moderne ? Redige une dissertation structuree.",
     _controle_riche),

    # ── factuel : pas de boucle, garde-fou honnete ──
    ("fact-def", "factuel", "Qu'est-ce que la photosynthese ?",
     lambda r, m: (_sans_boucle(r) and _sans_balise(r) and len(r) > 40,
                   f"{m['latence']:.1f}s")),
    ("fact-horsligne", "factuel", "Quel est le prix du carburant en France aujourd'hui ?",
     lambda r, m: (_sans_boucle(r), f"{m['latence']:.1f}s")),
]


def _executer(ia, question: str) -> tuple[str, dict]:
    t0 = time.time()
    r = ia.executer_detaille(question)
    meta = {"latence": round(time.time() - t0, 2),
            "cerveau": str(r.get("cerveau_choisi", "-")),
            "experts": sorted(r.get("experts") or set())}
    return str(r.get("reponse", "")), meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench_qualite")
    ap.add_argument("--rapide", action="store_true",
                    help="saute la categorie riche (2 min au lieu de 5)")
    ap.add_argument("--qid", metavar="ID", default="",
                    help="n'execute qu'une question (iteration rapide)")
    ap.add_argument("--garder-cache", action="store_true",
                    help="ne PAS effacer le cache du banc au demarrage")
    ap.add_argument("--json", metavar="CHEMIN", default="",
                    help="ecrit le detail complet en JSON")
    args = ap.parse_args(argv)

    # CACHE FRAIS : un run precedent aurait stocke une vieille generation
    # (parfois SALE — c'etait le cas de riche-puissance) que le run suivant
    # servirait en 0 ms, masquant la regression au lieu de la montrer.
    if not args.garder_cache and _CACHE_BANC.exists():
        _CACHE_BANC.unlink()
        print(f"[banc] cache precedent efface : {_CACHE_BANC.name}")

    from aura import config
    config.appliquer()
    from aura.orchestrateur import Aura1B

    ia = Aura1B()
    resultats = []
    print(f"{'id':20s} {'categorie':10s} {'ok':>3s}  mesure")
    print("-" * 74)
    for qid, cat, question, controle in QUESTIONS:
        if args.rapide and cat == "riche":
            continue
        if args.qid and qid != args.qid:
            continue
        reponse = ""
        try:
            reponse, meta = _executer(ia, question)
            ok, detail = controle(reponse, meta)          # type: ignore[operator]
        except Exception as e:                             # jamais de crash banc
            ok, detail = False, f"exception: {e}"
            meta = {"latence": 0.0, "cerveau": "?"}
        ligne = {"id": qid, "categorie": cat, "question": question,
                 "ok": bool(ok), "mesure": detail, **meta}
        if not ok and reponse:
            ligne["reponse"] = reponse        # diagnostic du KO (JSON)
        resultats.append(ligne)
        print(f"{qid:20s} {cat:10s} {'OK' if ok else 'KO':>3s}  {detail}")

    total = len(resultats)
    bons = sum(1 for r in resultats if r["ok"])
    print("-" * 74)
    print(f"SCORE : {bons}/{total}  ({100 * bons / max(total, 1):.0f} %)")
    for cat in sorted({r["categorie"] for r in resultats}):
        sous = [r for r in resultats if r["categorie"] == cat]
        n_ok = sum(1 for r in sous if r["ok"])
        print(f"    {cat:10s} {n_ok}/{len(sous)}")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"score": f"{bons}/{total}", "resultats": resultats},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {args.json}")
    return 0 if bons == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
