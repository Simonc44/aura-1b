"""Fenetre Aura style macOS : feux tricolores a gauche, apercu a gauche.

Barre de titre personnalisee (sans bordure Windows) : rouge = fermer,
jaune = reduire, vert = agrandir/restaurer ; fenetre deplaçable par la barre,
double-clic = zoom. Panneau Apercu a gauche (experts, cerveau, latence,
derniere reponse), conversation a droite.

Usage : python -m aura.gui [--smoke]   (--smoke : ferme seul apres 2 s)
"""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk

_ROUGE = "#FF5F57"
_JAUNE = "#FEBC2E"
_VERT = "#28C840"
_FOND_BARRE = "#2D2D2A"
_FOND_APP = "#1E1E1C"
_FOND_APERCU = "#252523"
_TEXTE = "#E8E6E3"
_TEXTE_FAIBLE = "#9A9891"
_ACCENT = "#7C9EF5"
_NL = chr(10)


class FenetreAura(tk.Tk):
    def __init__(self, smoke: bool = False) -> None:
        super().__init__()
        self.title("Aura-1B")
        self.configure(bg=_FOND_APP)
        self.overrideredirect(True)
        self._centrer()

        self._ia = None
        self._file: "queue.Queue" = queue.Queue()
        self._maximise = False
        self._geo_initiale = ""
        self._occupe = False
        self._glisser_origine: "tuple[int, int] | None" = None

        self._construire_barre()
        self._construire_corps()
        self._construire_pied()
        self._geo_initiale = self.geometry()

        threading.Thread(target=self._charger_ia, daemon=True).start()
        self.after(80, self._pomper_file)
        if smoke:
            self.after(2500, self.destroy)

    # ---------------- barre de titre macOS ----------------
    def _construire_barre(self) -> None:
        barre = tk.Frame(self, bg=_FOND_BARRE, height=38)
        barre.pack(side="top", fill="x")
        barre.pack_propagate(False)

        def bouton(parent, couleur, glyphe, commande):
            b = tk.Canvas(parent, width=13, height=13, bg=_FOND_BARRE,
                          highlightthickness=0, cursor="hand2")
            b.create_oval(1, 1, 12, 12, fill=couleur, outline="")
            b.create_text(6.5, 7, text=glyphe, fill="#5A2A20",
                          font=("Segoe UI", 8, "bold"))
            b.pack(side="left", padx=(10 if glyphe == chr(215) else 4, 0))
            b.bind("<Button-1>", lambda _e: commande())
            return b

        bouton(barre, _ROUGE, chr(215), self.destroy)
        bouton(barre, _JAUNE, chr(8722), self.iconify)
        bouton(barre, _VERT, chr(43), self._basculer_zoom)

        titre = tk.Label(barre, text="Aura-1B", bg=_FOND_BARRE,
                         fg=_TEXTE_FAIBLE, font=("Segoe UI", 10, "bold"))
        titre.pack(side="left", expand=True)

        self._etat = tk.Label(barre, text="chargement du cerveau...",
                              bg=_FOND_BARRE, fg=_TEXTE_FAIBLE,
                              font=("Segoe UI", 9))
        self._etat.pack(side="right", padx=10)

        for w in (barre, titre):
            w.bind("<Button-1>", self._debut_glisser)
            w.bind("<B1-Motion>", self._glisser)
        barre.bind("<Double-Button-1>", lambda _e: self._basculer_zoom())

    def _debut_glisser(self, event) -> None:
        self._glisser_origine = (event.x, event.y)

    def _glisser(self, event) -> None:
        if self._glisser_origine is None:
            return
        dx = event.x - self._glisser_origine[0]
        dy = event.y - self._glisser_origine[1]
        self.geometry(f"+{self.winfo_x() + dx}+{self.winfo_y() + dy}")

    def _basculer_zoom(self) -> None:
        if self._maximise:
            self.geometry(self._geo_initiale)
        else:
            self._geo_initiale = self.geometry()
            self.geometry(f"{self.winfo_screenwidth()}x"
                          f"{self.winfo_screenheight()}+0+0")
        self._maximise = not self._maximise

    def _centrer(self) -> None:
        self.update_idletasks()
        l, h = 980, 640
        x = max(0, (self.winfo_screenwidth() - l) // 3)
        y = max(0, (self.winfo_screenheight() - h) // 3)
        self.geometry(f"{l}x{h}+{x}+{y}")

    # ---------------- corps : apercu a gauche, chat a droite ----------------
    def _construire_corps(self) -> None:
        corps = tk.Frame(self, bg=_FOND_APP)
        corps.pack(side="top", fill="both", expand=True)

        apercu = tk.Frame(corps, bg=_FOND_APERCU, width=280)
        apercu.pack(side="left", fill="y", padx=(0, 2))
        apercu.pack_propagate(False)

        tk.Label(apercu, text="Apercu", bg=_FOND_APERCU, fg=_TEXTE_FAIBLE,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14,
                                                     pady=(14, 4))
        self._ap_lignes = tk.Label(apercu, text="En attente...",
                                   bg=_FOND_APERCU, fg=_TEXTE,
                                   font=("Consolas", 9), justify="left",
                                   anchor="nw", wraplength=250)
        self._ap_lignes.pack(anchor="nw", padx=14, pady=4, fill="x")

        tk.Label(apercu, text="Derniere reponse", bg=_FOND_APERCU,
                 fg=_TEXTE_FAIBLE, font=("Segoe UI", 10, "bold")
                 ).pack(anchor="w", padx=14, pady=(16, 4))
        self._ap_reponse = tk.Text(apercu, bg=_FOND_APERCU, fg=_TEXTE,
                                   wrap="word", relief="flat",
                                   font=("Segoe UI", 9), state="disabled")
        self._ap_reponse.pack(fill="both", expand=True, padx=10, pady=4)

        chat = tk.Frame(corps, bg=_FOND_APP)
        chat.pack(side="left", fill="both", expand=True)
        self._journal = tk.Text(chat, bg=_FOND_APP, fg=_TEXTE, wrap="word",
                                relief="flat", font=("Segoe UI", 10),
                                state="disabled", padx=16, pady=12)
        self._journal.pack(side="top", fill="both", expand=True)
        self._journal.tag_config("toi", foreground=_ACCENT,
                                 font=("Segoe UI", 10, "bold"))
        self._journal.tag_config("aura", foreground=_TEXTE)
        self._journal.tag_config("info", foreground=_TEXTE_FAIBLE,
                                 font=("Segoe UI", 9))

        pied_chat = tk.Frame(chat, bg=_FOND_APP)
        pied_chat.pack(side="bottom", fill="x", padx=12, pady=10)
        self._saisie = tk.Entry(pied_chat, bg=_FOND_APERCU, fg=_TEXTE,
                                relief="flat", font=("Segoe UI", 10),
                                insertbackground=_TEXTE)
        self._saisie.pack(side="left", fill="x", expand=True, ipady=7,
                          padx=(0, 8))
        self._saisie.bind("<Return>", lambda _e: self._envoyer())
        self._btn_envoyer = tk.Button(pied_chat, text="Envoyer",
                                      command=self._envoyer, bg=_ACCENT,
                                      fg="#10131A", relief="flat",
                                      font=("Segoe UI", 10, "bold"),
                                      padx=14, state="disabled")
        self._btn_envoyer.pack(side="right")

    def _construire_pied(self) -> None:
        pied = tk.Frame(self, bg=_FOND_BARRE, height=22)
        pied.pack(side="bottom", fill="x")
        tk.Label(pied, text="100 % local - Llama 3.2 1B + experts "
                 "deterministes", bg=_FOND_BARRE, fg=_TEXTE_FAIBLE,
                 font=("Segoe UI", 8)).pack(side="left", padx=10)

    # ---------------- cerveau + reponses en tache de fond ----------------
    def _charger_ia(self) -> None:
        t0 = time.time()
        try:
            from aura.orchestrateur import Aura1B
            ia = Aura1B()
            self._file.put(("pret", ia, round(time.time() - t0, 1)))
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", str(e)))

    def _envoyer(self) -> None:
        question = self._saisie.get().strip()
        if not question or self._ia is None or self._occupe:
            return
        self._occupe = True
        self._btn_envoyer.config(state="disabled")
        self._saisie.delete(0, "end")
        self._journal.config(state="normal")
        self._journal.insert("end", "toi  " + question + _NL + _NL, "toi")
        self._journal.config(state="disabled")
        self._journal.see("end")
        self._etat.config(text="reflexion...")
        threading.Thread(target=self._travailler, args=(question,),
                         daemon=True).start()

    def _travailler(self, question: str) -> None:
        t0 = time.time()
        ia = self._ia
        if ia is None:
            return
        try:
            r = ia.executer_detaille(question)
            self._file.put(("reponse", r, round(time.time() - t0, 1)))
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", str(e)))

    def _pomper_file(self) -> None:
        try:
            while True:
                msg = self._file.get_nowait()
                if msg[0] == "pret":
                    self._ia = msg[1]
                    self._occupe = False
                    self._btn_envoyer.config(state="normal")
                    self._etat.config(text="pret (" + str(msg[2]) + " s)")
                    self._ap_lignes.config(text="Cerveau charge." + _NL
                                           + "Posez votre question.")
                elif msg[0] == "reponse":
                    self._afficher(msg[1], msg[2])
                elif msg[0] == "erreur":
                    self._etat.config(text="erreur")
                    self._journal.config(state="normal")
                    self._journal.insert("end", "[erreur] " + msg[1]
                                         + _NL + _NL, "info")
                    self._journal.config(state="disabled")
                    self._occupe = False
                    self._btn_envoyer.config(state="normal")
        except queue.Empty:
            pass
        self.after(80, self._pomper_file)

    def _afficher(self, r: dict, dt: float) -> None:
        reponse = str(r.get("reponse", ""))
        experts = ", ".join(sorted(r.get("experts") or set())) or "-"
        cerveau = str(r.get("cerveau_choisi", "-"))
        self._journal.config(state="normal")
        self._journal.insert("end", "aura " + reponse + _NL + _NL, "aura")
        self._journal.config(state="disabled")
        self._journal.see("end")
        self._ap_lignes.config(text="Experts : " + experts + _NL
                               + "Cerveau : " + cerveau + _NL
                               + "Latence : " + str(dt) + " s")
        self._ap_reponse.config(state="normal")
        self._ap_reponse.delete("1.0", "end")
        self._ap_reponse.insert("1.0", reponse[:1500])
        self._ap_reponse.config(state="disabled")
        self._etat.config(text="pret")
        self._occupe = False
        self._btn_envoyer.config(state="normal")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="aura.gui")
    ap.add_argument("--smoke", action="store_true",
                    help="ferme seul apres 2 s (test)")
    args = ap.parse_args(argv)
    app = FenetreAura(smoke=args.smoke)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
