"""Soak test memoire : simule des heures de conversation en minutes.

Ce que les tests unitaires ne couvrent pas : la FENETRE glissante sur
des centaines de tours, l'ETAT evenementiel ecrase puis re-verifie, le
RLM sur un fichier d'historique deja gros, la ROTATION d'archives en
conditions reelles et la stabilite RAM (pas de croissance non bornee
des structures Python). Aucun LLM : le soak ne charge pas le cerveau,
il teste la memoire (memoire_conversation) — rapide et reproductible.

Chaque tour simule ~3,6 s de conversation reelle : 2000 tours ≈ 2 h.

Usage :
  uv run python scripts/soak_memoire.py                  # ~2 h simulees
  uv run python scripts/soak_memoire.py --tours 5000     # plus long
  uv run python scripts/soak_memoire.py --rapide         # ~1 min
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from aura.memoire_conversation import MemoireConversation  # noqa: E402
import aura.memoire_conversation as mc  # noqa: E402

_NOMS = ("Pierre", "Alice", "Karim", "Sophie", "Julien", "Nadia")
_BLAGUES = [
    "Qu'est-ce que tu penses de la récursivité ?",
    "Explique-moi la photosynthèse autrement.",
    "Raconte une blague sur les serveurs.",
    "Pourquoi le ciel est bleu ?",
    "Compare Python et le C++ pour un débutant.",
    "Rédige un paragraphe sur la patience.",
]


def _rss_mo() -> float | None:
    """RAM du processus (Mo). Best effort : None si non mesurable."""
    try:
        import psutil  # type: ignore[import-untyped]
        return float(psutil.Process().memory_info().rss) / 1e6
    except Exception:
        pass
    try:
        # Windows sans psutil : GetProcessMemoryInfo via ctypes
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p,
                                               ctypes.POINTER(_PMC),
                                               wintypes.DWORD)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                      ctypes.byref(pmc), pmc.cb):
            return float(pmc.WorkingSetSize) / 1e6
    except Exception:
        pass
    try:
        import resource        # Unix uniquement — garde-fou Windows ci-dessus
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1000  # type: ignore[attr-defined]
    except Exception:
        return None


def _p95(valeurs: list[float]) -> float:
    if not valeurs:
        return 0.0
    s = sorted(valeurs)
    return s[round(0.95 * (len(s) - 1))]


def _poser(memoire: MemoireConversation, role: str, texte: str) -> float:
    t0 = time.perf_counter()
    memoire.ajouter(role, texte)
    return (time.perf_counter() - t0) * 1000


def _phase_fidelite(tours: int) -> tuple[list[str], str]:
    """Phase A : RLM fidele sur un gros fichier (rotation desactivee).

    Codes rares disperses dans la conversation, noms qui changent :
    a la fin, TOUT doit rester retrouvable par l'outil RLM.
    """
    mc._TAILLE_ROTATION = 32 * 1024 * 1024          # rotation desactivee
    rnd = random.Random(42)
    codes: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        memoire = MemoireConversation(chemin=Path(tmp) / "conv.jsonl")
        nom = ""
        latences: list[float] = []
        for i in range(tours):
            if i % 40 == 0:
                nom = _NOMS[(i // 40) % len(_NOMS)]
                latences.append(_poser(memoire, "user", f"je m'appelle {nom}"))
                etat = memoire.etat()
                if etat.get("utilisateur") != nom:
                    return [], f"etat perdu au tour {i} : {etat} != {nom}"
            elif i % 97 == 0:
                code = f"SOAK-{i:04d}-XZ"
                codes.append(code)
                latences.append(_poser(memoire, "user",
                                       f"le code de la semaine est {code}."))
            else:
                latences.append(_poser(memoire, "user", rnd.choice(_BLAGUES)))
                _poser(memoire, "assistant", "tres bien, noté.")
            if i % 500 == 0:
                fen = memoire.fenetre()
                if len(fen) > 8:
                    return [], f"fenetre non bornee : {len(fen)} messages"

        taille = memoire._chemin.stat().st_size
        manquants = [c for c in codes if c not in memoire.rechercher(c)]
        if manquants:
            return [], f"RLM a perdu {len(manquants)}/{len(codes)} codes"
        print(f"[3/5] RLM fidelite      : OK "
              f"({len(codes)}/{len(codes)} codes retrouves, fichier {taille/1024:.0f} Ko)")
        print(f"      fenetre glissante : OK (bornee a 8 messages sur {tours} tours)")
        print(f"      etat evenementiel : OK ({tours // 40 + 1} noms re-verifies)")
        print(f"      latence ajouter   : p50 {statistics.median(latences):.2f} ms  "
              f"p95 {_p95(latences):.2f} ms  max {max(latences):.1f} ms")
    return codes, ""


def _phase_rotation(tours: int, seuil: int) -> str:
    """Phase B : rotation d'archives en conditions reelles (seuil force).

    Le seuil est abaisse artificiellement (32 Mo d'historique = des mois
    d'usage) pour declencher plusieurs rotations pendant la session.
    """
    mc._TAILLE_ROTATION = seuil
    rnd = random.Random(7)
    codes_recents: list[str] = []
    # la purge des vieilles archives est PAR CONCEPTION (max _MAX_ARCHIVES) :
    # la garantie verifiee est que la fenetre de retention
    # (courant + archives ≈ 4 x seuil) garde les codes RECENTS retrouvables.
    # Les codes de la derniere decile (~10 % des tours ≈ 2 x seuil au plus)
    # sont donc tenus d'etre retrouvables par l'outil RLM.
    borne = tours * 9 // 10
    with tempfile.TemporaryDirectory() as tmp:
        memoire = MemoireConversation(chemin=Path(tmp) / "conv.jsonl")
        t0 = time.perf_counter()
        for i in range(tours):
            if i == borne:
                codes_recents.clear()
            if i % 50 == 0:
                code = f"ROT-{i:04d}-KQ"
                if i >= borne:
                    codes_recents.append(code)
                memoire.ajouter("user", f"le mot de passe du wifi est {code}.")
            else:
                memoire.ajouter("user", rnd.choice(_BLAGUES))
                memoire.ajouter("assistant", "d'accord.")
            if i % 100 == 0:
                if len(memoire.fenetre()) > 8:
                    return f"fenetre non bornee apres rotation (tour {i})"
        duree = time.perf_counter() - t0

        archives = memoire._archives()
        taille_courant = (memoire._chemin.stat().st_size
                          if memoire._chemin.exists() else 0)
        if not archives:
            return "aucune rotation declenchee — seuil trop haut ?"
        if len(archives) > mc._MAX_ARCHIVES:
            return f"{len(archives)} archives conservees (max {mc._MAX_ARCHIVES})"
        if taille_courant >= seuil:
            return f"fichier courant {taille_courant} o >= seuil {seuil} o"
        manquants = [c for c in codes_recents
                     if c not in memoire.rechercher(c)]
        if manquants:
            return (f"RLM a perdu des codes recents apres rotation : "
                    f"{manquants[:3]}")
        print(f"[4/5] rotation archives : OK ({len(archives)} archive(s) conservee(s), "
              f"courant {taille_courant/1024:.0f} Ko < seuil {seuil/1024:.0f} Ko)")
        # les fichiers d'archive restent du JSONL valide (integralite)
        for a in archives:
            for ligne in a.read_text(encoding="utf-8",
                                     errors="ignore").splitlines():
                if ligne.strip():
                    json.loads(ligne)     # ValueError si archive corrompue
        print(f"      {tours} tours en {duree:.1f} s "
              f"({duree / tours * 1000:.2f} ms/tour), purge conforme "
              f"(max {mc._MAX_ARCHIVES} archives), archives JSONL valides")
    return ""


def soak(tours: int, seuil: int) -> int:
    print(f"[soak] {tours} tours (~{tours * 3.6 / 3600:.1f} h de conversation simulee), "
          f"rotation forcee a {seuil / 1024:.0f} Ko")
    rss_debut = _rss_mo()
    print("[1/5] demarrage phase A (fidelite RLM, sans rotation)...")

    codes, erreur = _phase_fidelite(tours)
    if erreur:
        print(f"ECHEC phase A : {erreur}")
        return 1

    print("[2/5] demarrage phase B (rotation d'archives, seuil force)...")
    erreur = _phase_rotation(max(tours // 2, 200), seuil)
    if erreur:
        print(f"ECHEC phase B : {erreur}")
        return 1

    rss_fin = _rss_mo()
    if rss_debut is not None and rss_fin is not None:
        croissance = rss_fin - rss_debut
        etat_ram = "OK" if croissance < 100 else "CROISSANCE SUSPECTE"
        print(f"[5/5] RAM              : {etat_ram} "
              f"({rss_debut:.0f} -> {rss_fin:.0f} Mo, croissance {croissance:+.0f} Mo)")
    else:
        print("[5/5] RAM              : non mesuree (psutil absent)")
    print("SOAK: PASS")
    return 0


def main() -> int:
    parseur = argparse.ArgumentParser(description="Soak test de la memoire Aura")
    parseur.add_argument("--tours", type=int, default=2000,
                         help="nombre de tours simules (defaut 2000 ~ 2 h)")
    parseur.add_argument("--seuil", type=int, default=16 * 1024,
                         help="seuil de rotation force pour la phase B (octets)")
    parseur.add_argument("--rapide", action="store_true",
                         help="version courte (~300 tours)")
    args = parseur.parse_args()
    tours = 300 if args.rapide else args.tours
    return soak(tours, args.seuil)


if __name__ == "__main__":
    raise SystemExit(main())
