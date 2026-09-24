"""Rotation du secret de chiffrement du .aef — la procedure sensible.

Le secret .secret_aef.txt derive la cle AES-256-GCM du paquet publie.
S'il fuit (depot git, capture d'ecran, machine partagee), le seul
correctif est une ROTATION : dechiffrer avec l'ancien secret, re-chiffrer
avec un nouveau, verifier, remplacer. Ce script automatise exactement
la procedure a risque, avec les verifications integrees qui manquent
quand on la fait a la main :

  1. le .enc est bien un conteneur AUAE ;
  2. l'ancien secret DECHIFFRE reellement le paquet (sinon : abandon) ;
  3. empreinte SHA-256 du .aef clair memorisee ;
  4. re-chiffrement avec un nouveau secret (secrets.token_urlsafe(24)) ;
  5. ROUND-TRIP : le nouveau secret redonne le MEME .aef (SHA-256 egal,
     magic AURA) — sinon rien n'est remplace ;
  6. remplacement ATOMIQUE (tmp + replace) du .enc et du .secret_aef.txt,
     apres copie de secours .bak des deux ;
  7. confirmation interactive avant tout ecrasement (--oui pour passer).

Le secret n'est JAMAIS affiche en clair (seule son empreinte l'est) et
n'est ni loggue ni passe en argument de ligne de commande visible.

Usage :
  uv run python scripts/tourner_secret.py                 # rotation interactive
  uv run python scripts/tourner_secret.py --oui           # sans confirmation
  uv run python scripts/tourner_secret.py --verifier      # teste seulement
                                                        # (dechiffrement+SHA)
"""
from __future__ import annotations

import argparse
import hashlib
import secrets
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from aura import chiffrement  # noqa: E402

CHEMIN_ENC = RACINE / "aura_system.aef.enc"
CHEMIN_SECRET = RACINE / ".secret_aef.txt"


def _empreinte(donnees: bytes) -> str:
    return hashlib.sha256(donnees).hexdigest()


def _lire_secret(chemin: Path) -> str:
    if not chemin.exists():
        raise SystemExit(f"[rotation] {chemin} introuvable : rien a faire.")
    secret = chemin.read_text(encoding="utf-8").strip()
    if not secret:
        raise SystemExit(f"[rotation] {chemin} vide : rien a faire.")
    return secret


def _sauvegarde(chemin: Path) -> Path:
    backup = chemin.with_suffix(chemin.suffix + ".bak")
    backup.write_bytes(chemin.read_bytes())
    return backup


def verifier(chemin_enc: Path = CHEMIN_ENC,
             chemin_secret: Path = CHEMIN_SECRET) -> dict:
    """Verifie que le secret dechiffre reellement le .enc publie.

    Renvoie {'ok': bool, 'sha': str, 'taille': int}. N'ecrit RIEN.
    """
    secret = _lire_secret(chemin_secret)
    if not chemin_enc.exists():
        raise SystemExit(f"[rotation] {chemin_enc} introuvable : rien a verifier.")
    enc = chemin_enc.read_bytes()
    if enc[:4] != chiffrement.MAGIC_CHIFFRE:
        raise SystemExit("[rotation] le fichier n'est pas un conteneur chiffre "
                         "AUAE : abandon.")
    try:
        clair = chiffrement.dechiffrer(enc, secret)   # ValueError si mauvais
    except ValueError as e:
        raise SystemExit(f"[rotation] dechiffrement impossible avec le secret "
                         f"actuel : {e}") from e
    if clair[:4] != b"AURA":
        raise SystemExit("[rotation] header AURA inattendu apres dechiffrement : "
                         "paquet incoherent, abandon.")
    return {"ok": True, "sha": _empreinte(clair), "taille": len(clair)}


def tourner(chemin_enc: Path = CHEMIN_ENC,
            chemin_secret: Path = CHEMIN_SECRET,
            nouveau: str | None = None,
            confirmer: bool = True) -> dict:
    """Rotation complete : dechiffre -> re-chiffre -> verifie -> remplace.

    `nouveau` : secret impose (tests) ; par defaut genere aleatoirement.
    `confirmer` : demande l'accord avant d'ecraser (desactive par --oui).
    Renvoie {'sha', 'ancienne_empreinte_secret', 'nouvelle_empreinte_secret'}.
    """
    ancien = _lire_secret(chemin_secret)
    enc = chemin_enc.read_bytes()
    if enc[:4] != chiffrement.MAGIC_CHIFFRE:
        raise SystemExit("[rotation] le fichier n'est pas un conteneur AUAE : "
                         "abandon.")

    print("[1/5] dechiffrement avec l'ancien secret...")
    try:
        clair = chiffrement.dechiffrer(enc, ancien)   # echoue si secret errone
    except ValueError as e:
        raise SystemExit("[rotation] l'ancien secret ne dechiffre PAS le .enc "
                         f"({e}) — rien n'a ete modifie, abandon.") from e
    del enc
    if clair[:4] != b"AURA":
        raise SystemExit("[rotation] header AURA inattendu : abandon.")
    sha = _empreinte(clair)
    print(f"      SHA-256 .aef : {sha[:16]}... ({len(clair)/1e6:.0f} Mo)")

    if not nouveau:
        nouveau = secrets.token_urlsafe(24)
    if nouveau == ancien:
        raise SystemExit("[rotation] le nouveau secret est identique a "
                         "l'ancien : abandon.")

    print("[2/5] re-chiffrement avec le nouveau secret...")
    nouvel_enc = chiffrement.chiffrer(clair, nouveau)
    del clair

    print("[3/5] verification round-trip (le nouveau secret doit redonner "
          "le MEME .aef)...")
    tmp = chemin_enc.with_suffix(".enc.tmp")
    tmp.write_bytes(nouvel_enc)
    del nouvel_enc
    controle = chiffrement.dechiffrer(tmp.read_bytes(), nouveau)
    if _empreinte(controle) != sha or controle[:4] != b"AURA":
        tmp.unlink(missing_ok=True)
        raise SystemExit("[rotation] round-trip ECHOUE : l'ancien .enc et "
                         "l'ancien secret sont INTACTS, rien n'a ete remplace.")
    del controle

    if confirmer:
        reponse = input("[4/5] Remplacer le .enc et le secret ? "
                        "tape 'OUI' pour confirmer : ").strip()
        if reponse != "OUI":
            tmp.unlink(missing_ok=True)
            print("[rotation] abandonne — rien n'a ete modifie.")
            raise SystemExit(1)

    print("[5/5] remplacement atomique (+ sauvegardes .bak)...")
    bak_enc = _sauvegarde(chemin_enc)
    bak_secret = _sauvegarde(chemin_secret)
    tmp.replace(chemin_enc)
    chemin_secret.write_text(nouveau + "\n", encoding="utf-8")

    print(f"      OK. sauvegardes : {bak_enc.name}, {bak_secret.name}")
    print("      RAPPEL : uploader le nouveau .enc sur Hugging Face et "
          "sauvegarder le nouveau secret (USB / gestionnaire de mdp).")
    return {"sha": sha,
            "ancienne_empreinte_secret": _empreinte(ancien.encode())[:12],
            "nouvelle_empreinte_secret": _empreinte(nouveau.encode())[:12]}


def main() -> int:
    parseur = argparse.ArgumentParser(
        description="Rotation du secret de chiffrement du .aef")
    parseur.add_argument("--verifier", action="store_true",
                         help="teste seulement que le secret dechiffre le .enc")
    parseur.add_argument("--oui", action="store_true",
                         help="sans confirmation interactive")
    args = parseur.parse_args()
    if args.verifier:
        r = verifier()
        print(f"VERIF: OK — SHA-256 .aef {r['sha'][:16]}... "
              f"({r['taille']/1e6:.0f} Mo)")
        return 0
    r = tourner(confirmer=not args.oui)
    print(f"ROTATION: OK — nouveau secret empreinte "
          f"{r['nouvelle_empreinte_secret']}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
