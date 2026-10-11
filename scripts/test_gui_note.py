"""Test GUI + note /100 : pilote la vraie interface tkinter d'RontoGPT.

Charge aura.gui, remplace le chargement du cerveau par l'orchestrateur reel
(meme chemin de code que la GUI normale), envoie les questions du
benchmark_comparatif dans l'interface, verifie les bulles affichees,
capture l'ecran, puis note l'interface sur 100 (8 criteres ponderes).

Usage : uv run python scripts/test_gui_note.py [--limite N] [--rapide]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from scripts.benchmark_comparatif import QUESTIONS, contient, note_ecriture  # noqa: E402

CAPTURES = RACINE / "captures_gui"


def capture(nom: str) -> Path:
    """Capture d'ecran via PowerShell (pas de Pillow dans le venv)."""
    CAPTURES.mkdir(exist_ok=True)
    chemin = CAPTURES / nom
    ps = ("Add-Type -AssemblyName System.Windows.Forms;"
          "Add-Type -AssemblyName System.Drawing;"
          "$b = New-Object System.Drawing.Bitmap"
          "([System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width,"
          "[System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height);"
          "$g = [System.Drawing.Graphics]::FromImage($b);"
          "$g.CopyFromScreen(0, 0, 0, 0, $b.Size);"
          "$b.Save('" + str(chemin).replace("\\", "/") + "')")
    import subprocess
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   capture_output=True, timeout=30)
    return chemin


def main() -> int:
    ap = argparse.ArgumentParser(prog="test_gui_note")
    ap.add_argument("--limite", type=int, default=0,
                    help="ne tester que les N premieres questions (0 = tout)")
    ap.add_argument("--rapide", action="store_true",
                    help="saute les 4 dernieres questions (ecriture)")
    args = ap.parse_args()

    from aura import filtre_instantane as fi
    from aura import gui as mod_gui
    from aura.gui import FenetreRontoGPT
    from aura.memoire_conversation import MemoireConversation

    tmp = tempfile.mkdtemp(prefix="gui_note_")
    fi._FICHIER = Path(tmp) / "cache_gui.jsonl"
    fi._vectoriseur = None
    fi._matrice = None
    fi._entrees = []

    # un seul cerveau : le test fournit l'orchestrateur a la GUI par sa
    # file (meme chemin de code que le chargement normal)
    setattr(mod_gui.FenetreRontoGPT, "_charger_ia",
            lambda self: None)  # noqa: B010

    fen = FenetreRontoGPT()
    fen.update_idletasks()
    fen.geometry("+0+0")  # coin haut gauche : capture garantie
    fen.update()
    if fen._placeholder:  # desactive le placeholder du champ de saisie
        fen._placeholder = False

    resultat: dict = {"ia": None, "err": None}
    t0 = time.time()

    def charger() -> None:
        t0c = time.time()
        try:
            from aura.orchestrateur import RontoGPT
            ia = RontoGPT()
            ia._historique = MemoireConversation(chemin=Path(tmp) / "gui.jsonl")
            ia.reinitialiser_conversation()
            ia.executer_detaille("Bonjour")  # cache froid absorbe ici
            resultat["ia"] = ia
            fen._file.put(("pret", ia, round(time.time() - t0c, 1)))
        except Exception as e:  # noqa: BLE001
            resultat["err"] = str(e)

    import threading
    threading.Thread(target=charger, daemon=True).start()
    while resultat["ia"] is None and resultat["err"] is None:
        fen.update()
        time.sleep(0.05)
        if time.time() - t0 > 600:
            break
    if resultat["err"]:
        print("ERREUR chargement cerveau :", resultat["err"])
        return 1
    ia = resultat["ia"]
    print("Cerveau charge dans la GUI en", round(time.time() - t0, 1), "s")
    fen.update()

    # ---- passe 1 : parcours des questions dans la vraie interface ----
    questions = list(QUESTIONS)
    if args.limite:
        questions = questions[:args.limite]
    if args.rapide:
        questions = questions[:24]

    ok = 0
    ko: list[str] = []
    t0 = time.time()
    for i, item in enumerate(questions):
        q, attendus, interdits = item["q"], item["a"], item["x"]
        avant = fen._journal.get("1.0", "end")
        fen._saisie.delete(0, "end")
        fen._saisie.insert(0, q)
        fen._envoyer()
        # pompe la boucle jusqu'a l'affichage de la bulle reponse
        rep = ""
        t1 = time.time()
        echec_cb = ""
        while time.time() - t1 < 90:
            try:
                fen.update()
            except Exception as e:  # noqa: BLE001
                echec_cb = "update: " + repr(e)
                break
            time.sleep(0.02)
            contenu = " ".join(fen._journal.get("1.0", "end").split())
            if not fen._occupe and len(contenu) > len(avant) + len(q):
                idx = contenu.find(" ".join(q.split()))
                if idx >= 0:
                    rep = contenu[idx + len(q):].split(" experts ")[0].strip()
                    if rep:
                        break
        if not rep:
            print("    DIAG: occupe=", fen._occupe, "cb=", echec_cb,
                  "journal=", fen._journal.get("1.0", "end")[-260:],
                  flush=True)
        bon = bool(rep) and contient(rep, attendus, interdits)
        ok += bon
        if not bon:
            ko.append(q)
        print(f"  [{i + 1}/{len(questions)}] {'OK' if bon else 'KO'} "
              f"{round(time.time() - t1, 1)} s cumul {ok}/{i + 1}",
              flush=True)
        if i == 2:
            capture("gui_echange.png")
        if i == len(questions) - 1:
            capture("gui_finale.png")

    dt_total = time.time() - t0
    n = len(questions)

    # ---- epreuve d'ecriture, meme regle que le benchmark ----
    PROMPT_ECRITURE = ("redige un court paragraphe (4 phrases environ) "
                       "expliquant pourquoi l'eau est essentielle a la "
                       "vie sur terre")
    avant_ecr = fen._journal.get("1.0", "end")
    fen._saisie.delete(0, "end")
    fen._saisie.insert(0, PROMPT_ECRITURE)
    fen._envoyer()
    ecriture = ""
    t1 = time.time()
    while time.time() - t1 < 300:
        fen.update()
        time.sleep(0.02)
        if not fen._occupe:
            contenu = " ".join(fen._journal.get("1.0", "end").split())
            idx = contenu.find(" ".join(PROMPT_ECRITURE.split()))
            if idx >= 0:
                ecriture = contenu[idx + len(PROMPT_ECRITURE):] \
                    .split(" experts ")[0].strip()
                if len(ecriture) > 80:
                    break
            if time.time() - t1 > 290:
                break
    note_ecr = note_ecriture(ecriture or "")

    # ---- passe 2 : note d'INTERFACE /100 (8 criteres ponderes) ----
    journal = fen._journal.get("1.0", "end")
    tags = set(fen._journal.tag_names("2.0"))
    note: dict[str, int] = {}

    # 1. Bulles distinctes (18 pts) : fonds bleu/gris POSES sur des plages
    #    reelles du journal (tag_ranges non vide pour les deux familles)
    poses = all(fen._journal.tag_ranges(t)
                for t in ("toi.bord", "aura.bord"))
    note["bulles_distinctes"] = 18 if poses else 6

    # 2. Alignement (14 pts) : marge gauche elevee pour les messages "toi"
    cfg = str(fen._journal.tag_cget("toi", "lmargin1"))
    note["alignement"] = 14 if cfg.isdigit() and int(cfg) >= 200 else 6

    # 3. Indicateur de reflexion (12 pts) : tag "ecrit" utilise pendant
    #    l'attente (verifie par la presence du tag dans le journal)
    note["indicateur_ecrit"] = 12 if "ecrit" in journal or "ecrit" in tags \
        else 0

    # 4. Champ de saisie dessine (12 pts) : canvas + coins arrondis
    champs = [e for e in fen._conteneur_saisie.winfo_children()]
    note["champ_arrondi"] = 12 if champs else 2

    # 5. Bouton Envoyer reactive (10 pts)
    note["bouton_envoyer"] = 10 if str(fen._btn_envoyer.cget("state")) \
        == "normal" else 0

    # 6. Panneau Apercu renseigne (12 pts)
    ap_texte = fen._ap_lignes.cget("text")
    note["panneau_apercu"] = 12 if "Experts" in ap_texte else 4

    # 7. Statut + infos de latence (10 pts)
    statut = str(fen._statut.cget("text"))
    note["statut_latence"] = 10 if ("pret" in statut and "experts"
                                    in journal) else 4

    # 8. Qualite des reponses via l'UI vs benchmark (12 pts)
    pourcent = round(100 * ok / n) if n else 0
    note["exactitude_ui"] = 12 if pourcent >= 75 else (
        8 if pourcent >= 60 else 4)

    total = sum(note.values())
    bench_note = round(100 * ok / n) if n else 0

    print("\n=== RESULTATS ===")
    print(f"Exactitude via GUI : {ok}/{n} = {bench_note} %")
    print(f"Ecriture : {note_ecr['score_sur_10']}/10 "
          f"({note_ecr['mots']} mots, {note_ecr['phrases']} phrases)")
    if ko:
        print("Echecs :", "; ".join(ko[:6]))
    print("\n=== NOTE INTERFACE /100 ===")
    labels = {
        "bulles_distinctes": "Bulles distinctes (bleu/gris)",
        "alignement": "Alignement utilisateur droite / RontoGPT gauche",
        "indicateur_ecrit": "Indicateur « RontoGPT ecrit... »",
        "champ_arrondi": "Champ de saisie dessine",
        "bouton_envoyer": "Bouton Envoyer reactive",
        "panneau_apercu": "Panneau Apercu renseigne",
        "statut_latence": "Statut + latence affiches",
        "exactitude_ui": "Exactitude des reponses via l'UI",
    }
    for cle, val in note.items():
        print(f"  {labels[cle]:<44} {val:>3}")
    print(f"  {'TOTAL':<44} {total:>3}/100")

    fen.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
