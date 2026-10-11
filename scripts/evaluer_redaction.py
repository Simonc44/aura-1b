"""Banc de redaction avec conservation des sorties et grille humaine."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import banc_100 as banc  # noqa: E402
import bench_qualite as bq  # noqa: E402


def _features(answer: str) -> dict:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", answer) if p.strip()]
    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", answer)
                 if s.strip()]
    normalized = [re.sub(r"\W+", " ", s, flags=re.UNICODE) for s in sentences]
    unique = len(set(normalized)) / max(len(normalized), 1)
    return {
        "characters": len(answer),
        "paragraphs": len(paragraphs),
        "sentences": len(sentences),
        "unique_sentence_ratio": round(unique, 3),
        "legacy_form_pass": not bq._defauts_riche(answer),
        "legacy_form_issues": bq._defauts_riche(answer),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--qid", default="", help="ex: riche-03")
    parser.add_argument("--sortie", default="resultats/redaction_baseline.jsonl")
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions doit etre >= 1")

    dataset_path = ROOT / "datasets" / "eval_redaction_v1.jsonl"
    rubric = {row["id"]: row for row in
              (json.loads(line) for line in dataset_path.read_text(
                  encoding="utf-8").splitlines() if line.strip())}
    questions = [(qid, question) for qid, category, question, _ in banc.QUESTIONS
                 if category == "riche" and (not args.qid or qid == args.qid)]
    if not questions:
        parser.error(f"aucune question riche pour --qid {args.qid!r}")

    cache = ROOT / f".redaction_cache_{os.getpid()}.jsonl"
    cache.touch(exist_ok=True)
    os.environ["AURA_FILTRE_CACHE"] = str(cache)
    from aura import config, filtre_instantane
    from aura.memoire_conversation import MemoireConversation
    from aura.orchestrateur import RontoGPT

    config.appliquer()
    filtre_instantane.enregistrer = lambda question, reponse: None
    filtre_instantane.mettre_a_jour = lambda question, reponse: False
    ia = RontoGPT()
    output = ROOT / args.sortie
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    total, legacy_pass = 0, 0
    try:
        with output.open("w", encoding="utf-8") as stream:
            for repetition in range(1, args.repetitions + 1):
                for qid, question in questions:
                    history = ROOT / f".redaction_mem_{os.getpid()}_{repetition}_{qid}.jsonl"
                    ia._historique = MemoireConversation(chemin=history)
                    t0 = time.perf_counter()
                    try:
                        answer, meta = bq._executer(ia, question)
                        error = None
                    except Exception as exc:  # keep partial evaluation usable
                        answer = ""
                        meta = {"latence": round(time.perf_counter() - t0, 2),
                                "cerveau": "?", "experts": []}
                        error = f"{type(exc).__name__}: {exc}"
                    finally:
                        for path in (history, history.with_suffix(".etat.json")):
                            try:
                                path.unlink()
                            except FileNotFoundError:
                                pass
                    row = {
                        "id": qid,
                        "repetition": repetition,
                        "question": question,
                        "attendus": rubric[qid.replace("riche-", "redaction-")]["attendus"],
                        "reponse": answer,
                        "features": _features(answer),
                        "latence_s": meta["latence"],
                        "cerveau": meta["cerveau"],
                        "experts": meta["experts"],
                        "score_humain": {k: None for k in
                            ("pertinence", "couverture", "exactitude", "structure", "clarte")},
                        "erreur": error,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    total += 1
                    legacy_pass += row["features"]["legacy_form_pass"]
                    print(f"{qid} r{repetition}: "
                          f"{'OK' if row['features']['legacy_form_pass'] else 'KO'}; "
                          f"{row['features']['characters']} caracteres; "
                          f"{meta['latence']:.1f}s", flush=True)
    finally:
        try:
            cache.unlink()
        except FileNotFoundError:
            pass

    print(f"Bilan forme historique: {legacy_pass}/{total}; "
          f"duree {time.time() - start:.1f}s; sortie {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
