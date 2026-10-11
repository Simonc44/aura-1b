"""Mesure RontoGPT complet sur GSM8K — « le pipeline rattrape-t-il le cerveau ? »

Le banc lm-eval (scripts/lm_eval_aura.py) mesure le CERVEAU NU : 0,16 en 5-shot
(strict-match) sur 100 questions. Ce script pose les MEMES 100 questions
(test[0:100], comme --limit 100 seed 0) a RontoGPT complet — solveurs symboliques
d'abord, cerveau en repli — via l'API /v1/ask, et rapporte les trois chiffres
que la comparaison exige :

  * justesse   — extraction identique au banc (strict-match puis
                 flexible-extract), pour que l'ecart mesure le SYSTEME et pas
                 le harnais ;
  * latence    — distribution reelle (mediane, p90, max), pas une moyenne qui
                 cache la queue ;
  * part generee — combien de questions ONT REELLEMENT fait tourner le 1B.

Ce dernier chiffre est mesure DEUX FOIS, independamment, parce qu'il porte
toute la conclusion :
  1. par etiquette — le champ `cerveau` vaut p.ex. `calcul-verbal` (couche
     exacte, aucun token) ou `gemma3-1b` / `def-web+1b` (le 1B a tourne) ;
  2. par le serveur — le compteur `id_task` de /slots sur le port 8757, lu
     avant et apres. C'est le seul temoin que mon propre code ne produit pas.
     (/metrics renvoie 501 sur ce build, /slots non.)
Si les deux divergent, c'est l'etiquette qu'il faut corriger, pas le serveur.

Le resultat est ecrit au fur et a mesure (JSONL) : un run long reste lisible
a mi-chemin, et interrompu il vaut encore quelque chose.

Usage :
    .venv/Scripts/python.exe scripts/mesure_gsm8k_aura.py
    .venv/Scripts/python.exe scripts/mesure_gsm8k_aura.py --limite 20
    .venv/Scripts/python.exe scripts/mesure_gsm8k_aura.py --deja-fait
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_RACINE = Path(__file__).resolve().parent.parent
_SORTIE = _RACINE / "resultats" / "gsm8k_aura"

_API = "http://127.0.0.1:8760/v1/ask"
_SLOTS = "http://127.0.0.1:8757/slots"

# Les deux filtres de lm-eval/tasks/gsm8k/gsm8k.yaml, recopies a l'identique.
_STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
_FLEXIBLE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

# Etiquettes de cerveau ou le modele 1B tourne vraiment. Heuristique — c'est
# pourquoi elle est confrontee au compteur du serveur (cf. `_generations`).
_MOTIF_GENERATIF = "1b"


def _normaliser(valeur: str | None) -> str:
    """Meme nettoyage que les regexes_to_ignore du banc : , $ et point final."""
    if not valeur:
        return ""
    return valeur.replace(",", "").replace("$", "").strip().rstrip(".")


def extraire(texte: str) -> dict:
    """(strict, flexible) — la premiere extraction qui trouve gagne, comme lm-eval."""
    m = _STRICT.search(texte)
    strict = _normaliser(m.group(1) if m else None)
    trouve = _FLEXIBLE.findall(texte)
    candidats = [a or b for a, b in trouve]
    flexible = _normaliser(candidats[-1]) if candidats else ""
    return {"strict": strict, "flexible": flexible}


def poser(question: str, base: str, timeout: float) -> dict:
    corps = json.dumps({"question": question}).encode()
    requete = urllib.request.Request(
        base, data=corps, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(requete, timeout=timeout) as r:
        donnees = json.loads(r.read().decode())
    donnees["_mur"] = round(time.time() - t0, 2)
    return donnees


def taches_slots(base: str = _SLOTS) -> dict[int, int] | None:
    """`id_task` par slot llama.cpp — le compteur de generations du serveur.

    Temoin exterieur : il ne depend d'aucun de mes classements. None si le
    serveur ne repond pas (le rapport le dira, il n'inventera pas 0).
    """
    try:
        with urllib.request.urlopen(base, timeout=10) as r:
            slots = json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return {int(s["id"]): int(s.get("id_task") or 0) for s in slots}


def generations_serveur(avant: dict[int, int] | None,
                        apres: dict[int, int] | None) -> int | None:
    """Le nombre de generations reellement servies pendant le run."""
    if not avant or not apres:
        return None
    return sum(max(0, apres.get(i, 0) - t) for i, t in avant.items())


def _questions(limite: int) -> list[tuple[str, str]]:
    """Les `limite` premieres questions du test GSM8K + leur reponse attendue.

    test[0:limite] — c'est l'ordre que lm-eval consomme avec --limit, donc les
    memes questions que le 0,16 du cerveau nu.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    paires: list[tuple[str, str]] = []
    for i in range(min(limite, len(ds))):
        ex = ds[i]
        m = _STRICT.search(ex["answer"])
        paires.append((ex["question"], _normaliser(m.group(1) if m else None)))
    return paires


def _percentile(valeurs: list[float], p: float) -> float:
    if not valeurs:
        return 0.0
    triees = sorted(valeurs)
    rang = min(len(triees) - 1, int(round(p / 100 * (len(triees) - 1))))
    return triees[rang]


def _resume(enregistrements: list[dict], gen_serveur: int | None) -> dict:
    """Le rapport — calcule sur ce qui a ete enregistre, rien d'autre."""
    faits = [e for e in enregistrements if e.get("ok") is not None]
    if not faits:
        return {"n": 0, "generations_serveur": gen_serveur}
    latences = [float(e["latence"]) for e in faits]
    generatifs = [e for e in faits
                  if _MOTIF_GENERATIF in str(e.get("cerveau", "")).lower()]
    return {
        "n": len(faits),
        "juste_strict": round(sum(e["ok_strict"] for e in faits) / len(faits), 4),
        "juste_flexible": round(sum(e["ok_flexible"] for e in faits) / len(faits), 4),
        "latence_mediane": round(statistics.median(latences), 2),
        "latence_moyenne": round(statistics.fmean(latences), 2),
        "latence_p90": round(_percentile(latences, 90), 2),
        "latence_max": round(max(latences), 2),
        "generatifs_etiquette": len(generatifs),
        "part_sans_generation": round(1 - len(generatifs) / len(faits), 4),
        "generations_serveur": gen_serveur,
        "erreurs_api": sum(1 for e in enregistrements if e.get("ok") is None),
    }


def _rapport(resume: dict, sortie: Path) -> None:
    if not resume.get("n"):
        print("Aucun enregistrement exploitable.")
        return
    print("\n" + "=" * 64)
    print(f"AURA COMPLET sur GSM8K — {resume['n']} questions")
    print("=" * 64)
    print(f"  justesse strict-match      : {resume['juste_strict']:.2f}"
          "   (cerveau nu : 0.16)")
    print(f"  justesse flexible-extract  : {resume['juste_flexible']:.2f}"
          "   (cerveau nu : 0.20)")
    print(f"  latence mediane / p90 / max: {resume['latence_mediane']}s / "
          f"{resume['latence_p90']}s / {resume['latence_max']}s")
    print(f"  sans generation (etiquette): {resume['part_sans_generation']:.2f}"
          f"  ({resume['n'] - resume['generatifs_etiquette']}/{resume['n']})")
    serveur = resume.get("generations_serveur")
    print(f"  generations (compteur 8757): "
          f"{serveur if serveur is not None else 'INDISPONIBLE'}")
    if serveur is not None:
        ecart = serveur - resume["generatifs_etiquette"]
        verdict = "concordent" if ecart == 0 else f"ECART {ecart:+d}"
        print(f"  -> les deux mesures {verdict}")
    print(f"  erreurs API                : {resume['erreurs_api']}")
    print(f"\n  detail : {sortie}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RontoGPT complet sur GSM8K")
    p.add_argument("--limite", type=int, default=100,
                   help="nombre de questions (defaut : 100, comme le banc)")
    p.add_argument("--api", default=_API, help="URL de /v1/ask")
    p.add_argument("--slots", default=_SLOTS, help="URL de /slots (compteur)")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="timeout par question (defaut : 600 s)")
    p.add_argument("--sortie", default=None, help="fichier JSONL de resultats")
    p.add_argument("--deja-fait", action="store_true",
                   help="ne pose que les questions absentes du JSONL")
    args = p.parse_args(argv)

    _SORTIE.mkdir(parents=True, exist_ok=True)
    sortie = Path(args.sortie) if args.sortie else _SORTIE / "gsm8k_aura.jsonl"

    assert extraire("il reste #### 42.")["strict"] == "42", "extraction cassee"
    assert extraire("so 2,000 in total")["flexible"] == "2000", "flexible casse"

    avant = taches_slots(args.slots)
    if avant is None:
        print("! /slots injoignable : la part generee ne sera pas contre-verifiee",
              file=sys.stderr)

    questions = _questions(args.limite)
    deja: dict[int, dict] = {}
    if sortie.exists():
        for ligne in sortie.read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                e = json.loads(ligne)
                deja[int(e["i"])] = e

    print(f"{len(questions)} questions | sortie {sortie}"
          f" | deja {len(deja)} en base")
    with sortie.open("a", encoding="utf-8") as f:
        for i, (question, attendu) in enumerate(questions):
            if args.deja_fait and i in deja:
                continue
            enregistrement: dict = {"i": i, "attendu": attendu}
            try:
                r = poser(question, args.api, args.timeout)
                texte = str(r.get("reponse") or "")
                ex = extraire(texte)
                enregistrement.update({
                    "ok": True,
                    "reponse": texte[:400],
                    "strict": ex["strict"],
                    "flexible": ex["flexible"],
                    "ok_strict": ex["strict"] == attendu,
                    "ok_flexible": ex["flexible"] == attendu,
                    "experts": r.get("experts") or [],
                    "cerveau": r.get("cerveau", "-"),
                    "latence": float(r.get("latence") or 0.0),
                    "mur": r.get("_mur", 0.0),
                })
            except (urllib.error.URLError, OSError, ValueError) as e:
                enregistrement.update({"ok": None, "erreur": str(e)[:200]})
            deja[i] = enregistrement
            f.write(json.dumps(enregistrement, ensure_ascii=False) + "\n")
            f.flush()
            etat = ("OK" if enregistrement.get("ok_strict") else
                    "--" if enregistrement.get("ok") else "!!")
            print(f"  [{i + 1}/{len(questions)}] {etat} "
                  f"lat={enregistrement.get('latence', '-')}s "
                  f"cerveau={enregistrement.get('cerveau', '-')}", flush=True)

    apres = taches_slots(args.slots)
    gen = generations_serveur(avant, apres)
    _rapport(_resume(list(deja.values()), gen), sortie)
    return 0


if __name__ == "__main__":
    sys.exit(main())
