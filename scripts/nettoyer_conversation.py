"""Nettoie l'historique de conversation des reponses POLLUEES.

Certaines anciennes reponses d'RontoGPT RECOPIAIENT le contexte injecte (brique
de style « Exemples de style attendu », exemple factuel « diffusion de
Rayleigh », consigne d'outil « Pour lire un fichier texte... », libelle
« SECTION A REDIGER »). Ces blocs etaient ensuite RESERVIS a des questions
sans rapport. Le correctif d'orchestrateur empeche la NOUVELLE pollution
(brique et consignes retirées sur les questions de raisonnement) ; ce script
retire la pollution DEJA stockee.

Securite : une sauvegarde horodatee est ecrite avant toute modification
(--essai ne fait que rapporter).

Usage :
    python scripts/nettoyer_conversation.py --essai
    python scripts/nettoyer_conversation.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
CONV = RACINE / "aura" / ".conversation.jsonl"

#: Fragments qui ne peuvent venir QUE d'une recopie de contexte injecte.
MARQUEURS = (
    "suivant les consignes",
    "exemples de style attendu",
    "diffusion de rayleigh",
    "rempart contre l'imprevu",
    "voici une réponse à votre question",
    "structure de la réponse",
    "pour lire un fichier texte",
    "section a rediger",
    "redige ta réponse",
    "plan a suivre",
)


def est_polluee(contenu: str) -> bool:
    """La reponse recopie-t-elle un morceau de contexte injecte ?"""
    c = (contenu or "").lower()
    return any(m in c for m in MARQUEURS)


def nettoyer(lignes: list[str]) -> tuple[list[str], int]:
    """Retire les tours pollues : la reponse ET la question qui la precede.

    Renvoie (lignes gardees, nombre de reponses polluees retirees). Une ligne
    illisible (JSON casse) est CONSERVEE : on ne detruit pas par accident.
    """
    gardees: list[str] = []
    retires = 0
    for ligne in lignes:
        try:
            d = json.loads(ligne)
        except Exception:
            gardees.append(ligne)
            continue
        if d.get("role") == "assistant" and est_polluee(d.get("content", "")):
            if gardees:
                try:
                    precedent = json.loads(gardees[-1])
                except Exception:
                    precedent = {}
                if precedent.get("role") == "user":
                    gardees.pop()
            retires += 1
            continue
        gardees.append(ligne)
    return gardees, retires


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nettoyer_conversation")
    ap.add_argument("--fichier", default=str(CONV))
    ap.add_argument("--essai", action="store_true", help="ne rien ecrire")
    args = ap.parse_args(argv)

    fichier = Path(args.fichier)
    if not fichier.is_file():
        print("aucun historique :", fichier)
        return 0
    lignes = fichier.read_text(encoding="utf-8").splitlines()
    gardees, retires = nettoyer(lignes)
    print(f"{len(lignes)} lignes -> {len(gardees)} "
          f"({retires} reponses polluees retirees)")
    if args.essai or retires == 0:
        print("(rien ecrit)" if args.essai else "rien a nettoyer.")
        return 0
    sauvegarde = fichier.with_suffix(
        fichier.suffix + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
    sauvegarde.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    fichier.write_text("\n".join(gardees) + "\n", encoding="utf-8")
    print("sauvegarde :", sauvegarde.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
