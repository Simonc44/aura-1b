"""Benchmark comparatif : Aura (hybride) vs LLM seul vs concurrent HF.

But : repondre a la question "Aura est-il aussi puissant qu'un modele plus
gros ?" avec un POURCENTAGE chiffre, sur les memes questions fermees posees
a chaque modele, dans les memes conditions (temp 0, reponse courte).

Bras :
  aura : le pipeline complet (orchestrateur + experts deterministes)
  llama: Llama 3.2 1B Q4_K_M seul, meme moteur llama.cpp, meme machine
  qwen : Qwen 2.5 1.5B Instruct (via Ollama local, base r1-u300-lite)

Dataset : 28 questions fermes a reponse courte verifiable (math, langue,
logique, savoir, francais-typo) + 1 epreuve d'ecriture notee a part.

Usage :
  uv run python scripts/benchmark_comparatif.py --bras aura
  uv run python scripts/benchmark_comparatif.py --bras llama
  uv run python scripts/benchmark_comparatif.py --bras qwen
  uv run python scripts/benchmark_comparatif.py --rapport bench_comparatif.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
import time
from typing import Any
import unicodedata
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Normalisation et verification des reponses
# ---------------------------------------------------------------------------

def _sans_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _norm(s: str) -> str:
    s = _sans_accents(s).lower()
    s = s.replace("\u2019", "'").replace("\u00a0", " ")
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    s = re.sub(r"(?<=\d) (?=\d)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def contient(reponse: str, attendus: list[str], interdits: list[str]) -> bool:
    r = _norm(reponse)
    ok = all(_norm(a) in r for a in attendus)
    ko = not any(_norm(i) in r for i in interdits) if interdits else True
    return ok and ko


# ---------------------------------------------------------------------------
# Dataset : questions fermes (reponse courte, verification automatique)
# ---------------------------------------------------------------------------

QUESTIONS: list[dict] = [
    # --- math (10) ----------------------------------------------------------
    {"cat": "math", "q": "combien font 17 fois 23 ?",
     "a": ["391"], "x": []},
    {"cat": "math", "q": "combien font 144 divise par 12 ?",
     "a": ["12"], "x": []},
    {"cat": "math", "q": "combien font 7 plus 8 fois 3 ?",
     "a": ["31"], "x": ["45"]},
    {"cat": "math", "q": "combien font racine carree de 169 ?",
     "a": ["13"], "x": []},
    {"cat": "math", "q": "combien font 2 puissance 10 ?",
     "a": ["1024"], "x": []},
    {"cat": "math", "q": "quel est le PGCD de 48 et 36 ?",
     "a": ["12"], "x": []},
    {"cat": "math", "q": "combien font 15 pourcent de 200 ?",
     "a": ["30"], "x": []},
    {"cat": "math", "q": "j'ai 15 euros, j'en depense 7, combien il m'en reste ?",
     "a": ["8"], "x": []},
    {"cat": "math", "q": "combien font 45 multiplie par 8 ?",
     "a": ["360"], "x": []},
    {"cat": "math", "q": "combien font 6 puissance 3 ?",
     "a": ["216"], "x": []},
    # --- langue (6) : conjugaisons et definitions connues du systeme --------
    {"cat": "langue", "q": "conjugue faire au futur a la premiere personne du pluriel",
     "a": ["ferons"], "x": []},
    {"cat": "langue", "q": "conjugue pouvoir au present a la troisieme personne du singulier",
     "a": ["peut"], "x": []},
    {"cat": "langue", "q": "conjugue avoir a l'imparfait a la troisieme personne du singulier",
     "a": ["avait"], "x": []},
    {"cat": "langue", "q": "conjugue savoir au futur a la premiere personne du singulier",
     "a": ["saurai"], "x": []},
    {"cat": "langue", "q": "qu'est-ce que la photosynthese ?",
     "a": ["lumiere"], "x": []},
    {"cat": "langue", "q": "qu'est-ce qu'un algorithme ?",
     "a": ["etapes"], "x": []},
    # --- logique (4) ----------------------------------------------------------
    {"cat": "logique", "q": "tous les chats sont des felins, felix est un chat : felix est-il un felin ? reponds oui ou non.",
     "a": ["oui"], "x": ["non"]},
    {"cat": "logique", "q": "je suis plus grand que marie, marie est plus grande que lea : qui est le plus grand entre moi et lea ?",
     "a": ["moi"], "x": ["lea"]},
    {"cat": "logique", "q": "dans une course, paul est devant pierre, pierre est devant jacques : qui est dernier ?",
     "a": ["jacques"], "x": ["paul", "pierre"]},
    {"cat": "logique", "q": "si tous les A sont des B et aucun B n'est un C, peut-on dire qu'aucun A n'est un C ? reponds oui ou non.",
     "a": ["oui"], "x": ["non"]},
    # --- savoir (4) -----------------------------------------------------------
    {"cat": "savoir", "q": "quelle est la capitale de la france ?",
     "a": ["paris"], "x": []},
    {"cat": "savoir", "q": "combien de planetes dans le systeme solaire ?",
     "a": ["8", "huit"], "x": ["9", "neuf"]},
    {"cat": "savoir", "q": "qui a ecrit les miserables ?",
     "a": ["hugo"], "x": []},
    {"cat": "savoir", "q": "quel gaz les plantes absorbent-elles pour la photosynthese ?",
     "a": ["co2", "dioxyde"], "x": ["oxygene"]},
    # --- francais-typo (4) : la reponse doit etre un francais POLI ------------
    {"cat": "francais-typo", "q": "conjugue parler au present a la premiere personne du pluriel",
     "a": ["parlons"], "x": []},
    {"cat": "francais-typo", "q": "qu'est-ce que la gravitation ?",
     "a": ["attire", "force"], "x": []},
    {"cat": "francais-typo", "q": "qu'est-ce qu'une metaphore ?",
     "a": ["figure", "style"], "x": []},
    {"cat": "francais-typo", "q": "conjugue venir au futur a la troisieme personne du pluriel",
     "a": ["viendront"], "x": []},
]

# Epreuve d'ecriture : PAS comptee dans le pourcentage, notee a part /10.
SUJET_ECRITURE = ("redige un court paragraphe (4 phrases environ) expliquant "
                  "pourquoi l'eau est essentielle a la vie sur terre")


def note_ecriture(texte: str) -> dict:
    """Note deterministe sur 10 : phrases, longueur, majuscules, ponctuation,
    mots de liaison. Mesure la FORME, pas les connaissances."""
    mots = len(re.findall(r"[a-zA-Z'\u00e0-\u00ff\u0100-\u017f]+", texte))
    phrases = [p for p in re.split(r"[.!?]+", texte) if p.strip()]
    n_phr = len(phrases)
    debut_maj = sum(1 for p in phrases if p.strip()[:1].isupper())
    fin_ponct = 1 if texte.strip()[-1:] in ".!?" else 0
    nrm = _norm(texte)
    liaisons = sum(1 for m in ("parce", "donc", "ainsi", "en effet", "de plus",
                               "c'est pourquoi", "sans elle") if m in nrm)
    score = 0
    score += 3 if 3 <= n_phr <= 6 else (1 if n_phr > 0 else 0)
    score += 2 if 40 <= mots <= 160 else (1 if mots >= 15 else 0)
    score += 2 if debut_maj >= min(n_phr, 2) else 0
    score += 1 if fin_ponct else 0
    score += 2 if liaisons >= 1 else 0
    return {"score_sur_10": min(score, 10), "mots": mots, "phrases": n_phr,
            "liaisons": liaisons}


# ---------------------------------------------------------------------------
# Bras 1 : Aura complet (pipeline local)
# ---------------------------------------------------------------------------

def bras_aura() -> tuple:
    from aura import filtre_instantane as fi
    from aura.orchestrateur import Aura1B
    from aura.memoire_conversation import MemoireConversation

    # demarrage A FROID : cache de reponses vierge (les LLM concurrents
    # n'ont pas de cache ; on mesure la qualite intrinseque du systeme)
    tmp = tempfile.mkdtemp(prefix="bench_aura_")
    fi._FICHIER = Path(tmp) / "cache_bench.jsonl"
    fi._vectoriseur = None
    fi._matrice = None
    fi._entrees = []
    ia = Aura1B()
    ia._historique = MemoireConversation(chemin=Path(tmp) / "bench.jsonl")
    ia.reinitialiser_conversation()
    return ia.executer, ia.reinitialiser_conversation


# ---------------------------------------------------------------------------
# Bras 2 : Llama 1B seul (meme moteur, aucun expert Aura)
# ---------------------------------------------------------------------------

def bras_llama_seul() -> tuple:
    from llama_cpp import Llama
    from aura import llama_cerveau

    llm = Llama(model_path=llama_cerveau._CHEMIN_GGUF,
                n_ctx=1536, n_batch=768, n_threads=6, n_threads_batch=6,
                flash_attn=True, type_k=8, type_v=8, n_gpu_layers=0,
                verbose=False)
    sys_prompt = ("Tu reponds en francais, de facon tres courte et directe, "
                  "a la question de l'utilisateur.")

    def executer(question: str, X=None, y=None) -> str:
        out: Any = llm.create_chat_completion(
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": question}],
            max_tokens=120, temperature=0.0)
        return out["choices"][0]["message"]["content"] or ""

    return executer, lambda: None


# ---------------------------------------------------------------------------
# Bras 3 : concurrent Qwen 2.5 1.5B via Ollama local
# ---------------------------------------------------------------------------

def bras_qwen_ollama(modele: str = "r1-u300-lite:latest") -> tuple:
    import requests

    url = "http://localhost:11434/api/chat"

    def executer(question: str, X=None, y=None) -> str:
        r = requests.post(url, json={
            "model": modele, "stream": False,
            "messages": [{"role": "user", "content": question}],
            "options": {"temperature": 0.0, "num_predict": 120},
        }, timeout=600)
        r.raise_for_status()
        return str(r.json()["message"]["content"]).strip()

    return executer, lambda: None


# ---------------------------------------------------------------------------
# Execution d'un bras
# ---------------------------------------------------------------------------

def passer_questions(executer, reinit, epreuve_ecriture: bool) -> dict:
    resultats = []
    for i, item in enumerate(QUESTIONS):
        t0 = time.time()
        try:
            rep = executer(item["q"])
        except Exception as e:  # noqa: BLE001
            rep = f"__ERREUR__ {e}"
        dt = time.time() - t0
        ok = contient(rep, item["a"], item["x"])
        resultats.append({"cat": item["cat"], "q": item["q"], "ok": ok,
                          "latence_s": round(dt, 3), "extrait": rep[:110]})
        print(f"  [{i + 1:>2}/{len(QUESTIONS)}] {'OK' if ok else 'KO'} "
              f"{dt:6.1f}s  {item['cat']}", flush=True)
        reinit()
    ecriture = None
    if epreuve_ecriture:
        t0 = time.time()
        try:
            rep = executer(SUJET_ECRITURE)
        except Exception as e:  # noqa: BLE001
            rep = f"__ERREUR__ {e}"
        dt = time.time() - t0
        ecriture = {**note_ecriture(rep), "latence_s": round(dt, 1),
                    "texte": rep[:600]}
        print(f"  ecriture : {ecriture['score_sur_10']}/10 "
              f"({ecriture['mots']} mots, {dt:.0f}s)", flush=True)
    return {"questions": resultats, "ecriture": ecriture}


def resume(bres: dict) -> dict:
    qs = bres["questions"]
    cats: dict = {}
    for c in sorted({q["cat"] for q in qs}):
        sous = [q for q in qs if q["cat"] == c]
        cats[c] = {"ok": sum(1 for q in sous if q["ok"]), "total": len(sous),
                   "pct": round(100 * sum(1 for q in sous if q["ok"]) / len(sous))}
    total_ok = sum(1 for q in qs if q["ok"])
    lat = [q["latence_s"] for q in qs]
    res: dict = {"categories": cats,
                 "global_pct": round(100 * total_ok / len(qs)),
                 "global_ok": f"{total_ok}/{len(qs)}",
                 "p50_s": round(statistics.median(lat), 3),
                 "p95_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 3)}
    if bres["ecriture"]:
        res["ecriture_sur_10"] = bres["ecriture"]["score_sur_10"]
        res["ecriture_mots"] = bres["ecriture"]["mots"]
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bras", default="aura",
                    help="aura, llama, qwen (separes par des virgules)")
    ap.add_argument("--rapport", default=None)
    args = ap.parse_args()

    rap: dict = {"dataset": len(QUESTIONS), "bras": {}}
    chemin_rapport = Path(args.rapport) if args.rapport else None
    if chemin_rapport and chemin_rapport.exists():
        try:
            precedent = json.loads(chemin_rapport.read_text(encoding="utf-8"))
            bras_prec = precedent.get("bras", {})
            rap["bras"].update(bras_prec)
            print(f"  (rapport precedent charge : {', '.join(bras_prec) or 'vide'})",
                  flush=True)
        except Exception:
            pass
    for nom in [b.strip() for b in args.bras.split(",") if b.strip()]:
        print(f"\n=== BRAS {nom.upper()} ===", flush=True)
        if nom == "aura":
            exe, rei = bras_aura()
        elif nom == "llama":
            exe, rei = bras_llama_seul()
        elif nom == "qwen":
            exe, rei = bras_qwen_ollama()
        else:
            raise SystemExit(f"bras inconnu : {nom}")
        bres = passer_questions(exe, rei, epreuve_ecriture=True)
        rap["bras"][nom] = {**resume(bres), "detail": bres["questions"],
                            "ecriture": bres["ecriture"]}
        r = rap["bras"][nom]
        print(f"  GLOBAL : {r['global_ok']} = {r['global_pct']} %   "
              f"p50 {r['p50_s']} s / p95 {r['p95_s']} s", flush=True)
        for c, v in r["categories"].items():
            print(f"    {c:<14} {v['ok']}/{v['total']} = {v['pct']} %", flush=True)
        if r.get("ecriture_sur_10") is not None:
            print(f"    ecriture      {r['ecriture_sur_10']}/10 "
                  f"({r['ecriture_mots']} mots)", flush=True)

    if "aura" in rap["bras"]:
        a = rap["bras"]["aura"]
        print("\n=== VERDICT ===", flush=True)
        for autre in ("llama", "qwen"):
            if autre not in rap["bras"]:
                continue
            b = rap["bras"][autre]
            gagne = a["global_pct"] - b["global_pct"]
            print(f"  aura {a['global_pct']} %  vs  {autre} {b['global_pct']} %  "
                  f"({gagne:+d} pts)   ecriture {a.get('ecriture_sur_10')}/10 vs "
                  f"{b.get('ecriture_sur_10')}/10", flush=True)
            if b["p50_s"] > 0:
                print(f"    latence p50 : {a['p50_s']} s vs {b['p50_s']} s "
                      f"(x{b['p50_s'] / a['p50_s']:.0f} plus rapide)", flush=True)

    if args.rapport:
        Path(args.rapport).write_text(
            json.dumps(rap, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\n  rapport -> {args.rapport}", flush=True)


if __name__ == "__main__":
    main()
