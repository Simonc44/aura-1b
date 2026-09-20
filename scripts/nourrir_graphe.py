"""Nourrit le graphe de faits d'Aura depuis knowledge.jsonl (R1-U300).

Chaque entree {cat, q, a} devient un triplet du graphe :
    sujet   = la question (les mots de la question servent d'index)
    relation = la categorie (« cuisine », « sante (info generale) »...)
    objet   = la reponse (tronquee a MAX_OBJET caracteres)

Regles de securite :
- categorie « identite » EXCLUE : ces entrees decrivent R1-U300, pas Aura
  (sinon Aura se presenterait comme un autre assistant) ;
- « sante » et « droit » marquees « (info generale) » : le graphe ne doit
  jamais suggerer un avis professionnel ;
- deduplication exacte par graphe_faits.ajouter → re-run sans doublon.

Usage :
    python scripts/nourrir_graphe.py [chemin_knowledge.jsonl]
"""
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from aura import graphe_faits  # noqa: E402

SOURCE_PAR_DEFAUT = Path(r"C:/Users/admin/Desktop/Dev/IA/R1-U300/knowledge.jsonl")
EXCLUES = {"identite"}
INFO_GENERALE = {"sante", "droit"}
MAX_OBJET = 400


def nourrir(chemin: Path) -> dict:
    """Convertit les entrees en triplets. Renvoie les compteurs."""
    ajoutes, ignorees, exclus = 0, 0, 0
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                e = json.loads(ligne)
            except json.JSONDecodeError:
                ignorees += 1
                continue
            cat = (e.get("cat") or "").strip()
            q = (e.get("q") or "").strip()
            a = (e.get("a") or "").strip()
            if cat in EXCLUES:
                exclus += 1
                continue
            if not q or not a:
                ignorees += 1
                continue
            relation = f"{cat} (info generale)" if cat in INFO_GENERALE else cat
            if graphe_faits.ajouter(q, relation, a[:MAX_OBJET],
                                    source=f"knowledge_{cat or 'divers'}"):
                ajoutes += 1
            else:
                ignorees += 1
    return {"ajoutes": ajoutes, "ignorees": ignorees, "exclus": exclus,
            "total": len(graphe_faits._triplets)}


def main() -> int:
    chemin = Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE_PAR_DEFAUT
    if not chemin.exists():
        print(f"[!] introuvable : {chemin}")
        return 1
    r = nourrir(chemin)
    print(f"[graphe] ajoutes={r['ajoutes']}  deja-presents/invalides="
          f"{r['ignorees']}  exclus(identite)={r['exclus']}  "
          f"total triplets={r['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
