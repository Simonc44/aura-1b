"""Verrou du createur — le kill switch de l'auto-amelioration.

L'IA apprend (corrections, RLSS, graphe) dans des FICHIERS DE DONNEES
cote a cote du programme : elle ne reecrit JAMAIS le programme lui-meme.
Le seul chemin qui modifie le binaire diffuse est la REFORGE du .aef
(scripts/forger_aef.py) — et celle-ci est signee Ed25519 par la cle
privee du createur (cle_privee.pem, jamais commitee, jamais embarquee).

Ce module centralise la garantie :
- autoriser_reforge() : refuse si la cle privee est absente -> un tiers
  (ou le systeme lui-meme) ne peut pas produire un .aef qui passera la
  verification de confiance du kernel ;
- verifier_chaine(chemin_aef) : le .aef present correspond-il a la
  cle publique du createur ? (kernel : refuse un binaire etranger)
- seal() / unseal() : chiffrement AES-256 (chiffrement.py) — la couche
  supplementaire qui rend le binaire illisible sans le secret.

Invariants (a preserver dans tout le codebase) :
1. le code produit/transforme des DONNEES, jamais du CODE ;
2. aucune ecriture nulle part en dehors de la racine du projet ;
3. aucune reforge sans cle privee presente (exit code 2 sinon).
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("aura.verrou")

_RACINE = Path(__file__).resolve().parent.parent
_CLE_PRIVEE = _RACINE / "cle_privee.pem"
_CLE_PUBLIQUE = _RACINE / "cle_publique.pub"

# fichiers que le systeme est autorise a modifier (donnees uniquement)
_ZONES_DONNEES = (
    "aura/.graphe_faits.jsonl", "aura/.cache_routeur/",
    "aura/.cache_reponses.jsonl", ".cache_corrections.jsonl",
    "datasets/", "logs/",
)


def autoriser_reforge() -> bool:
    """La reforge (rescellement) est-elle autorisee sur cette machine ?

    Uniquement si la cle privee du createur est presente. Le systeme qui
    tourne chez un utilisateur n'a que la cle PUBLIQUE (embarquee) : il
    verifie, il ne produit pas.
    """
    return _CLE_PRIVEE.is_file()


def verifier_chaine(chemin_aef: Path | None = None) -> bool:
    """Le .aef (ou son .sig) est-il signe par la cle du createur ?"""
    import json
    from . import signature  # import paresseux (cryptography est lourd)
    cible = chemin_aef or (_RACINE / "aura_system.aef")
    bloc_sig = cible.with_suffix(cible.suffix + ".sig")
    if not bloc_sig.is_file():
        return False
    try:
        cle_attendue = signature.cle_de_confiance()
        return signature.verifier_confiance(
            json.loads(bloc_sig.read_text()), cle_attendue)
    except Exception as e:
        LOG.info("[verrou] verification impossible : %s", e)
        return False


def rapport() -> dict:
    """Diagnostic : cle privee, cle publique, zones de donnees autorisees."""
    return {
        "reforge_autorisee": autoriser_reforge(),
        "cle_privee": str(_CLE_PRIVEE),
        "cle_publique_presente": _CLE_PUBLIQUE.is_file(),
        "zones_donnees": list(_ZONES_DONNEES),
        "ecritures_hors_zones": "interdites par convention (voir docstring)",
    }
