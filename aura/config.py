"""Configuration centralisee : aura.toml + profils (rapide / equilibre / qualite).

Pourquoi un module dedie : chaque module d'Aura lit ses variables ``AURA_*``
au moment de l'IMPORT (llama_cerveau._CHEMIN_GGUF, orchestrateur._SEUIL_*
...). ``appliquer()`` doit donc tourner AVANT d'importer l'orchestrateur —
c'est le cas dans ``aura.api``, ``aura.gui_web`` et ``aura.__main__``.

Priorite (de la plus forte a la plus faible) :
    1. variable d'environnement deja posee par l'utilisateur
    2. valeur du fichier aura.toml (racine du projet)
    3. valeurs du profil selectionne
    4. defauts du code (inchanges)

Usage :
    python -m aura.config            # affiche la configuration effective
    appliquer()                      # au demarrage d'un point d'entree
"""
from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

_RACINE = Path(__file__).resolve().parent.parent
_CHEMIN = _RACINE / "aura.toml"

# Profils : valeurs posees SEULEMENT si l'utilisateur ne les a pas posees.
# Rapide = on coupe tout ce qui n'est pas la generation (agents = plan +
# double passe + compresseurs, verification web) et on resserre le budget.
PROFILS: dict[str, dict[str, str]] = {
    "rapide": {"AURA_AGENTS": "0", "AURA_VERIF_WEB": "0",
               "AURA_BUDGET_S": "15"},
    "equilibre": {},
    "qualite": {"AURA_AGENTS": "1", "AURA_VERIF_WEB": "1",
                "AURA_BUDGET_S": "180"},
}

# Cle TOML -> variable d'environnement. Les booleens deviennent "1"/"0",
# ce qui colle aux conventions existantes (AURA_X != 0 = actif,
# AURA_X == 0 = coupe) pour toutes les cibles de cette table.
_CLES: dict[tuple[str, str], str] = {
    ("profil", "nom"): "AURA_PROFIL",
    ("llama", "gguf"): "AURA_GGUF",
    ("llama", "chemin"): "AURA_GGUF_CHEMIN",
    ("llama", "gpu"): "AURA_GPU",
    ("serveur", "actif"): "AURA_SERVEUR",
    ("serveur", "adresse"): "AURA_SERVEUR_ADRESSE",
    # AURA_SERVEUR_SPEC = speculative decoding COTE SERVEUR (rentable :
    # x2 mesure sur texte repetitif). AURA_SPEC (in-process) reste OFF par
    # defaut : la il ralentit (logits_all impose par llama_cpp).
    ("serveur", "spec"): "AURA_SERVEUR_SPEC",
    ("qualite", "budget_s"): "AURA_BUDGET_S",
    ("qualite", "verif_web"): "AURA_VERIF_WEB",
    ("qualite", "agents"): "AURA_AGENTS",
    ("qualite", "seuil_confiance"): "AURA_SEUIL_CONFIANCE",
    ("api", "port"): "AURA_API_PORT",
    ("maj", "verifier"): "AURA_MAJ",
    # Briques vectorielles (embeddings MiniLM) : encodeur partage,
    # detecteur de concepts, 2e avis flou du routeur.
    ("minilm", "actif"): "AURA_MINILM",
    ("concepts", "actif"): "AURA_CONCEPTS",
    ("concepts", "seuil"): "AURA_CONCEPTS_SEUIL",
    ("flou", "actif"): "AURA_FLOU",
}


def _lire_toml(chemin: Path | None = None) -> dict:
    """Fichier de configuration, ou {} (absent/corrompu = defauts du code)."""
    c = chemin or _CHEMIN
    try:
        if c.is_file():
            with open(c, "rb") as f:
                return tomllib.load(f)
    except Exception:
        pass
    return {}


def _vers_env(valeur) -> str:
    if isinstance(valeur, bool):
        return "1" if valeur else "0"
    return str(valeur)


def appliquer(chemin: Path | None = None,
              profil: str | None = None) -> dict[str, str]:
    """Pose les variables AURA_* du fichier + du profil. Renvoie ce qui a
    ete pose (pour affichage/logs). Les variables deja presentes dans
    l'environnement ne sont JAMAIS ecrasees (priorite utilisateur)."""
    deja = set(os.environ)                      # snapshot = choix utilisateur
    fichier = _lire_toml(chemin)
    pose: dict[str, str] = {}

    # 1) profil : argument > fichier > env > defaut
    nom = (profil
           or str(((fichier.get("profil") or {}).get("nom")) or "")
           or os.environ.get("AURA_PROFIL", "")
           or "equilibre").strip().lower()
    if nom not in PROFILS:
        nom = "equilibre"
    pose["AURA_PROFIL"] = nom
    for k, v in PROFILS[nom].items():
        pose[k] = v

    # 2) valeurs explicites du fichier (elles battent le profil)
    for (section, cle), env in _CLES.items():
        bloc = fichier.get(section)
        if isinstance(bloc, dict) and cle in bloc:
            pose[env] = _vers_env(bloc[cle])

    for k, v in pose.items():
        if k not in deja:
            os.environ[k] = v
    return {k: os.environ.get(k, "") for k in pose}


def efficace() -> dict[str, str]:
    """Les variables AURA_* actuellement visibles (affichage / diagnostic)."""
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("AURA_")}


def version() -> str:
    """Version du projet, lue dans pyproject.toml (source de verite)."""
    try:
        with open(_RACINE / "pyproject.toml", "rb") as f:
            donnees = tomllib.load(f)
        # version sous [project] (PEP 621) ; fallback sur un toplevel plat
        v = (donnees.get("project") or {}).get("version") or donnees.get("version")
        if v:
            return str(v)
    except Exception:
        pass
    return "0.0.0"


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="aura.config", description="affiche la configuration effective")
    ap.add_argument("--json", action="store_true", help="sortie JSON")
    ap.add_argument("--profil", default=None,
                    choices=sorted(PROFILS), help="forcer un profil (test)")
    args = ap.parse_args(argv)
    pose = appliquer(profil=args.profil)
    if args.json:
        print(json.dumps({"version": version(), "pose": pose,
                          "environnement": efficace()},
                         ensure_ascii=False, indent=2))
        return 0
    print(f"Aura {version()} — profil : {os.environ.get('AURA_PROFIL', '?')}")
    print(f"Fichier : {_CHEMIN if _lire_toml() else _CHEMIN.name + ' (absent -> defauts)'}")
    for k, v in efficace().items():
        print(f"  {k:26s} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
