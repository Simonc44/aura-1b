"""Harnais d'evaluation PERSONNEL d'Aura : mesure le SYSTEME, pas le cerveau.

Contrairement a lm-evaluation-harness (qui note un GGUF nu sur des taches
genériques), ce harnais evalue les VRAIS points d'entree d'Aura :

  --cible pipeline : le code source (aura.orchestrateur.Aura1B) in-process
  --cible aef      : le PAQUET SCELLE — kernel.boot() verifie (SHA-256 +
                     Ed25519), dechiffre (AURA_SECRET), extrait — puis c'est
                     le CODE DU PAQUET qui repond (ce qui est publie)
  --cible exe      : Aura.exe reel, un sous-processus par question — le
                     chemin utilisateur complet, du double-clic a la reponse

Dataset deterministe (aucun reseau requis) : maths niveau 0, calcul
verbal, identite constante, tri transitif, memoire multi-tours. Les
faits web sont exclus (non deterministes hors-ligne) : passer
--dataset mon_fichier.jsonl pour etendre.

Format dataset (JSONL) : {"question", "attendu", "mode":
"numerique"|"contient"|"exact"|"aucun", "categorie", "sequence"}
- sequence >= 1 : tours d'une meme conversation (l'etat memoire se
  construit puis est verifie) ; sequence absent/0 = tour isole.
- mode "aucun" : tour de construction d'etat, non compte dans le score.

Usage :
  uv run python scripts/harnais_evaluation.py                     # pipeline
  uv run python scripts/harnais_evaluation.py --cible aef         # paquet scelle
  uv run python scripts/harnais_evaluation.py --cible exe         # Aura.exe
  uv run python scripts/harnais_evaluation.py --rapport eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

# ── dataset integre (deterministe, hors-ligne) ─────────────────────────

DATASET: list[dict[str, Any]] = [
    # maths exactes (niveau 0 : AST, jamais de LLM)
    {"question": "combien font 7 fois 8", "attendu": "56",
     "mode": "numerique", "categorie": "math"},
    {"question": "calcule 12 plus 30", "attendu": "42",
     "mode": "numerique", "categorie": "math"},
    {"question": "combien font 9 fois 9", "attendu": "81",
     "mode": "numerique", "categorie": "math"},
    {"question": "calcule 100 moins 45", "attendu": "55",
     "mode": "numerique", "categorie": "math"},
    # calcul verbal (expert determinant)
    {"question": "j'ai 3 pommes, j'en mange 1, combien il m'en reste t il",
     "attendu": "2", "mode": "numerique", "categorie": "calcul-verbal"},
    {"question": "j'ai 10 euros, j'en depense 4, combien il me reste",
     "attendu": "6", "mode": "numerique", "categorie": "calcul-verbal"},
    # identite constante (hors des poids, 0 hallucination possible)
    {"question": "qui es tu", "attendu": "Aura",
     "mode": "contient", "categorie": "identite"},
    {"question": "qui t'a cree", "attendu": "Simon",
     "mode": "contient", "categorie": "identite"},
    # tri transitif (mini-SAT determinant)
    {"question": "Paul est plus age que Anna. Anna est plus age que Clara. "
                 "Qui est le plus age des trois ?",
     "attendu": "Paul", "mode": "contient", "categorie": "logique"},
    {"question": "Leo est plus lent que Mia. Mia est plus lente que Tom. "
                 "Qui est le plus rapide ?",
     "attendu": "Tom", "mode": "contient", "categorie": "logique"},
    # memoire multi-tours (etat evenementiel) : une conversation = memes
    # tours de MEME sequence (le 1er construit l'etat, le 2nd verifie)
    {"sequence": 1, "question": "je m'appelle Frederique",
     "mode": "aucun", "categorie": "memoire"},
    {"sequence": 1, "question": "comment je m'appelle",
     "attendu": "Frederique", "mode": "contient", "categorie": "memoire"},
    {"sequence": 2, "question": "j'ai achete 7 serveurs",
     "mode": "aucun", "categorie": "memoire"},
    {"sequence": 2, "question": "combien j'ai de serveurs",
     "attendu": "7", "mode": "numerique", "categorie": "memoire"},
    # langue symbolique (fast-lang : conjugaison et definitions O(1))
    {"question": "conjugue etre au futur", "attendu": "serai",
     "mode": "contient", "categorie": "langue"},
    {"question": "conjugue parler au present", "attendu": "parlons",
     "mode": "contient", "categorie": "langue"},
    {"question": "qu'est-ce que la photosynthese", "attendu": "lumiere",
     "mode": "contient", "categorie": "langue"},
]

# ── verification ────────────────────────────────────────────────────────

_RE_NOMBRE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _normaliser(texte: str) -> str:
    """Minuscule, sans accents, espaces compacts."""
    n = unicodedata.normalize("NFKD", (texte or "").lower())
    sans = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(sans.split())


def verifier(reponse: str, attendu: str, mode: str) -> bool:
    """Verifie une reponse selon le mode du dataset."""
    if mode == "aucun":
        return True
    if mode == "numerique":
        nombres = _RE_NOMBRE.findall(reponse or "")
        if not nombres:
            return False
        attendu_n = float(attendu.replace(",", "."))
        return any(float(n.replace(",", ".")) == attendu_n for n in nombres)
    if mode == "contient":
        return _normaliser(attendu) in _normaliser(reponse)
    if mode == "exact":
        return _normaliser(attendu) == _normaliser(reponse)
    raise ValueError(f"mode inconnu : {mode}")


# ── cibles d'evaluation ────────────────────────────────────────────────
# chaque cible renvoie (poser, reinitialiser, nettoyer)

def _cible_pipeline():
    """Code source, in-process. Memoire redirigee vers un fichier temporaire
    (le harnais ne pollue jamais la vraie .conversation.jsonl)."""
    from aura.orchestrateur import Aura1B
    from aura.memoire_conversation import MemoireConversation

    ia = Aura1B()
    tmp = tempfile.TemporaryDirectory()
    ia._historique = MemoireConversation(
        chemin=Path(tmp.name) / "eval.jsonl")
    return ia.executer, ia.reinitialiser_conversation, tmp


def _cible_aef(chemin_aef: Path):
    """Paquet scelle : boot() verifie/dechiffre/extrait puis c'est le CODE
    DU PAQUET qui repond. MemoireConversation importee depuis le code boote
    (pas du source) : on evalue exactement ce qui voyage dans le .aef."""
    from aura import kernel

    if chemin_aef.suffix == ".enc" and not os.environ.get("AURA_SECRET"):
        f_secret = RACINE / ".secret_aef.txt"
        if f_secret.exists():
            os.environ["AURA_SECRET"] = f_secret.read_text(
                encoding="utf-8").strip()
    sysm = kernel.boot(chemin_aef)
    ia = sysm["module"].Aura1B()
    mc = sysm["module"].memoire_conversation      # version du PAQUET
    tmp = tempfile.TemporaryDirectory()
    ia._historique = mc.MemoireConversation(
        chemin=Path(tmp.name) / "eval.jsonl")
    return ia.executer, ia.reinitialiser_conversation, tmp


def _cible_exe(exe: Path, chemin_aef: Path):
    """Aura.exe reel : un sous-processus par question (chemin utilisateur).
    Le cache kernel rend le boot rapide apres la premiere verification.
    Limite structurelle : l'etat evenementiel est en RAM, chaque
    sous-processus demarre neuf — les tests memoire passent alors par le
    fichier RLM persistant (outil « cherche dans l'historique »), ou
    echouent honnetement : c'est une vraie mesure du chemin utilisateur."""
    if not exe.exists():
        raise SystemExit(f"[harnais] {exe} introuvable — construis-le : "
                         "uv run python scripts/construire_exe.py")
    env = dict(os.environ)
    if chemin_aef.suffix == ".enc" and not env.get("AURA_SECRET"):
        f_secret = RACINE / ".secret_aef.txt"
        if f_secret.exists():
            env["AURA_SECRET"] = f_secret.read_text(encoding="utf-8").strip()

    def poser(question: str) -> str:
        r = subprocess.run([str(exe), str(chemin_aef), question],
                           capture_output=True, text=True, timeout=600,
                           env=env)
        return (r.stdout or "").strip()

    return poser, (lambda: None), None


def charger_dataset(chemin: str | None) -> list[dict]:
    if not chemin:
        return [dict(d) for d in DATASET]
    items = []
    for ligne in Path(chemin).read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if ligne:
            items.append(json.loads(ligne))
    if not items:
        raise SystemExit(f"[harnais] dataset vide : {chemin}")
    return items


def _p95(valeurs: list[float]) -> float:
    s = sorted(valeurs)
    return s[round(0.95 * (len(s) - 1))] if s else 0.0


def evaluer(poser, reinitialiser, dataset: list[dict]) -> dict:
    """Fait tourner le dataset et renvoie le rapport (score, categories,
    latence). Les tours d'une meme sequence partagent la conversation ;
    un reset isole chaque bloc de sequence / question."""
    resultats: dict[str, dict] = {}
    latences: list[float] = []
    seq_precedente: int | None = None
    for item in dataset:
        seq = item.get("sequence") or 0
        if seq == 0 or seq != seq_precedente:
            reinitialiser()               # conversation isolee par bloc
        seq_precedente = seq
        t0 = time.perf_counter()
        reponse = poser(item["question"])
        latence = (time.perf_counter() - t0) * 1000
        latences.append(latence)
        mode = item.get("mode", "aucun")
        ok = verifier(reponse, item.get("attendu", ""), mode)
        cat = item.get("categorie", "divers")
        r = resultats.setdefault(cat, {"passe": 0, "total": 0, "echecs": []})
        if mode != "aucun":
            r["total"] += 1
            if ok:
                r["passe"] += 1
            else:
                r["echecs"].append({"question": item["question"],
                                    "attendu": item.get("attendu"),
                                    "reponse": (reponse or "")[:160]})
    total = sum(r["total"] for r in resultats.values())
    passes = sum(r["passe"] for r in resultats.values())
    return {"score": passes / total if total else 0.0, "passes": passes,
            "total": total, "categories": resultats,
            "latence_p50_ms": statistics.median(latences) if latences else 0.0,
            "latence_p95_ms": _p95(latences)}


def afficher(rapport: dict, cible: str) -> None:
    print(f"\n{'=' * 62}\nHARNAIS AURA — cible : {cible}\n{'=' * 62}")
    for cat, r in sorted(rapport["categories"].items()):
        etat = "OK " if r["passe"] == r["total"] else "!! "
        print(f"  {etat}{cat:<16} {r['passe']}/{r['total']}")
        for e in r["echecs"]:
            print(f"       echec : {e['question'][:50]}")
            print(f"         attendu {e['attendu']!r} — recu "
                  f"{e['reponse'][:80]!r}")
    print("-" * 62)
    print(f"  SCORE GLOBAL : {rapport['passes']}/{rapport['total']} "
          f"= {rapport['score'] * 100:.0f} %   "
          f"latence p50 {rapport['latence_p50_ms']:.0f} ms / "
          f"p95 {rapport['latence_p95_ms']:.0f} ms")


def main() -> int:
    parseur = argparse.ArgumentParser(
        description="Harnais d'evaluation du systeme Aura (pipeline/aef/exe)")
    parseur.add_argument("--cible", choices=("pipeline", "aef", "exe"),
                         default="pipeline")
    parseur.add_argument("--aef", default="aura_system.aef.enc",
                         help="chemin du paquet (cibles aef/exe)")
    parseur.add_argument("--exe", default="dist/Aura.exe",
                         help="chemin de l'exe (cible exe)")
    parseur.add_argument("--dataset", default=None,
                         help="JSONL supplementaire (defaut : dataset integre)")
    parseur.add_argument("--seuil", type=float, default=0.80,
                         help="score minimal pour code de sortie 0")
    parseur.add_argument("--rapport", default=None,
                         help="ecrit le rapport JSON dans ce fichier")
    args = parseur.parse_args()

    dataset = charger_dataset(args.dataset)
    if args.cible == "pipeline":
        poser, reinitialiser, garder = _cible_pipeline()
    elif args.cible == "aef":
        poser, reinitialiser, garder = _cible_aef(RACINE / args.aef)
    else:
        poser, reinitialiser, garder = _cible_exe(RACINE / args.exe,
                                                  RACINE / args.aef)

    print(f"[harnais] {len(dataset)} questions, cible {args.cible}...")
    rapport = evaluer(poser, reinitialiser, dataset)
    afficher(rapport, args.cible)
    if garder is not None:
        garder.cleanup()
    if args.rapport:
        Path(args.rapport).write_text(
            json.dumps({"cible": args.cible, **rapport},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  rapport -> {args.rapport}")
    return 0 if rapport["score"] >= args.seuil else 1


if __name__ == "__main__":
    raise SystemExit(main())
