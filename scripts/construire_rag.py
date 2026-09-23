"""Construit rag.bin : l'index binaire consulte avant le web.

Sources (dedupliquees, tout ce qui est VERIFIE dans le systeme) :
  - datasets/{web,general,code}.jsonl : paires (question, reponse)
  - aura/.graphe_faits.jsonl: triplets (s, r, o) -> reponses composees
  - knowledge.jsonl         : base de savoir personnelle (si present)

PAS datasets/math.jsonl : les maths exactes appartiennent au niveau 0 et
au PGS (reponse calculee, jamais devinee). Un RAG statistique qui repond
« 12 x 12 = 37 » parce qu'une paire ressemble serait une regression.

Usage :
  python scripts/construire_rag.py            # reconstruit rag.bin
  python scripts/construire_rag.py --pack     # + regenere aura/_rag_data.py
                                              #   (embedded -> vit dans le .aef)
Le fichier est volontairement regime : seules les sources VERIFIEES y
entrent (experts deterministes, triplets web verifies, savoir manual).
"""
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from aura import rag_binaire  # noqa: E402

SORTIE = RACINE / "aura" / "rag.bin"
PACK = RACINE / "aura" / "_rag_data.py"


def _docs_datasets() -> list[dict]:
    docs = []
    dossier = RACINE / "datasets"
    if not dossier.exists():
        return docs
    for chemin in sorted(dossier.glob("*.jsonl")):
        if chemin.stem == "math":       # maths = domaine exact (niveau 0/PGS)
            continue
        tag = chemin.stem
        for ligne in chemin.read_text(encoding="utf-8", errors="ignore").splitlines():
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                obj = json.loads(ligne)
            except json.JSONDecodeError:
                continue
            q = (obj.get("question") or "").strip()
            r = (obj.get("reponse") or obj.get("answer") or "").strip()
            if q and r and len(q) >= 6:
                docs.append({"q": q, "a": r, "src": "dataset", "tag": tag})
    return docs


def _docs_graphe() -> list[dict]:
    docs = []
    chemin = RACINE / "aura" / ".graphe_faits.jsonl"
    if not chemin.exists():
        return docs
    for ligne in chemin.read_text(encoding="utf-8", errors="ignore").splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        try:
            t = json.loads(ligne)
        except json.JSONDecodeError:
            continue
        s, r, o = (t.get("s") or "").strip(), (t.get("r") or "").strip(), \
                  (t.get("o") or "").strip()
        if s and o:
            # question composee + reponse directe : « Paris capitale de ? »
            docs.append({"q": f"{s} {r}", "a": o,
                         "src": "graphe", "tag": "fait"})
            docs.append({"q": f"quel est le {r} {s}", "a": o,
                         "src": "graphe", "tag": "fait"})
    return docs


def _docs_knowledge() -> list[dict]:
    docs = []
    for nom in ("knowledge.jsonl", "aura/knowledge.jsonl"):
        chemin = RACINE / nom
        if not chemin.exists():
            continue
        for ligne in chemin.read_text(encoding="utf-8", errors="ignore").splitlines():
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                obj = json.loads(ligne)
            except json.JSONDecodeError:
                continue
            q = (obj.get("question") or obj.get("q") or obj.get("titre")
                 or "").strip()
            r = (obj.get("reponse") or obj.get("a") or obj.get("texte")
                 or obj.get("contenu") or "").strip()
            if q and r:
                docs.append({"q": q, "a": r[:600],
                             "src": "knowledge", "tag": "savoir"})
    return docs


def _dedupliquer(docs: list[dict]) -> list[dict]:
    vus: set[str] = set()
    uniques = []
    for d in docs:
        cle = rag_binaire.tokeniser(d["q"])
        cle = " ".join(sorted(cle))[:120]
        if cle in vus:
            continue
        vus.add(cle)
        uniques.append(d)
    return uniques


def construire(pack: bool = False) -> int:
    docs = _dedupliquer(_docs_datasets() + _docs_graphe() + _docs_knowledge())
    if not docs:
        print("[rag] aucune source trouvee — rien a construire")
        return 1

    import lzma
    import struct
    bloc_docs = lzma.compress(json.dumps(docs, ensure_ascii=False).encode("utf-8"),
                              preset=6)
    en_tete = struct.pack("<4sHII", rag_binaire.MAGIQUE,
                          rag_binaire.VERSION, len(docs),
                          4 + struct.calcsize("<HII") + len(docs) * 64)
    with open(SORTIE, "wb") as f:
        f.write(en_tete)
        for d in docs:
            f.write(rag_binaire.signature_minhash(
                rag_binaire.tokeniser(d["q"])))
        f.write(bloc_docs)

    taille = SORTIE.stat().st_size
    print(f"[rag] {len(docs)} documents -> {SORTIE.name} ({taille/1024:.0f} Ko)")

    if pack:
        avec = "BINAIRE = (\n"
        brut = SORTIE.read_bytes()
        # chunks pour rester lisible et compilable
        morceaux = [brut[i:i + 96] for i in range(0, len(brut), 96)]
        avec += "".join(
            f"    {m!r}\n" for m in morceaux)
        avec += ")\n"
        PACK.write_text(
            '"""Index RAG embarque (genere par scripts/construire_rag.py --pack).\n\n'
            'Present dans le code = present dans le .aef : le savoir verifie\n'
            'voyage avec le paquet scelle, consultable hors-ligne en ms.\n"""\n'
            + avec, encoding="utf-8")
        print(f"[rag] embarque -> {PACK.relative_to(RACINE)} "
              f"({PACK.stat().st_size/1024:.0f} Ko)")
    return 0


if __name__ == "__main__":
    raise SystemExit(construire(pack="--pack" in sys.argv))
