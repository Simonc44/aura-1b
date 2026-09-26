"""Aura-Vocal : assistant vocal flottant style TypeFlux, moteur Aura.

Inspire de https://github.com/mylxsw/typeflux (UX « maintenir pour
parler »), moteur de reponse = Aura-1B (Llama 3.2 1B + experts
deterministes), reconnaissance vocale = faster-whisper (SYSTRAN) 100 %
local, voix de synthese Windows via pyttsx3.

Maintenir le bouton pour parler (ou la touche F9) : au relachement la
question est transcrite, Aura repond, la reponse est dite a voix haute
et copiee dans le presse-papiers (colle-la ou tu veux, comme TypeFlux).

Usage :
  uv run python aura_vocal.py [--sans-voix] [--smoke]
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Any

# couleurs du theme sombre macOS, comme la GUI fenetree
_ROUGE = "#FF5F57"
_JAUNE = "#FEBC2E"
_VERT = "#28C840"
_FOND = "#1E1E1C"
_FOND_PASTILLE = "#2D2D2A"
_TEXTE = "#E8E6E3"
_TEXTE_FAIBLE = "#9A9891"
_ACCENT = "#7C9EF5"
_NL = chr(10)
GUIL = chr(171) + " " + chr(187)


class Pastille(tk.Tk):
    """Fenetre flottante compacte : pastille micro + bulle de reponse."""

    def __init__(self, avec_voix: bool = True, smoke: bool = False) -> None:
        super().__init__()
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(bg=_FOND_PASTILLE)
        self._construire()
        self._placer_bas_droite()

        self.avec_voix = avec_voix
        self._file: "queue.Queue" = queue.Queue()
        self._occupe = False
        self._ecoute = False
        self._ia: Any = None
        self._whisper: Any = None
        self._tts: Any = None

        threading.Thread(target=self._charger, daemon=True).start()
        self.after(100, self._pomper)
        self.bind("<F9>", self._toggle_global)
        if smoke:
            self.after(4000, self.destroy)

    # ---------------- UI ----------------
    def _construire(self) -> None:
        bandeau = tk.Frame(self, bg=_FOND_PASTILLE)
        bandeau.pack(fill="x")
        self._lab_titre = tk.Label(
            bandeau, text=" AURA  " + chr(183) + "  vocal ",
            bg=_FOND_PASTILLE, fg=_TEXTE_FAIBLE,
            font=("Segoe UI", 9, "bold"))
        self._lab_titre.pack(side="left", padx=6, pady=4)
        self._lab_etat = tk.Label(bandeau, text="chargement...",
                                  bg=_FOND_PASTILLE, fg=_TEXTE_FAIBLE,
                                  font=("Segoe UI", 8))
        self._lab_etat.pack(side="right", padx=6)
        self._lab_titre.bind("<Button-1>", self._debut_glisser)
        self._lab_titre.bind("<B1-Motion>", self._glisser)

        corps = tk.Frame(self, bg=_FOND_PASTILLE)
        corps.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._rond = tk.Canvas(corps, width=44, height=44,
                               bg=_FOND_PASTILLE, highlightthickness=0,
                               cursor="hand2")
        self._rond.pack(side="left")
        self._dessine_rond(_ACCENT)
        self._rond.bind("<ButtonPress-1>", self._debut_parler)
        self._rond.bind("<ButtonRelease-1>", self._fin_parler)

        self._bulle = tk.Label(
            corps, text=GUIL + " Maintiens le rond (ou F9), parle, "
            "relache." + GUIL + _NL + "Reponse dite a voix haute et "
            "copiee au presse-papiers.", bg=_FOND_PASTILLE, fg=_TEXTE,
            font=("Segoe UI", 9), justify="left", anchor="w",
            wraplength=300)
        self._bulle.pack(side="left", fill="x", expand=True, padx=8)

    def _dessine_rond(self, couleur: str) -> None:
        c = self._rond
        c.delete("all")
        c.create_oval(4, 4, 40, 40, fill=couleur, outline="")
        c.create_text(22, 22, text="mic" if False else chr(127917),
                      font=("Segoe UI Emoji", 13), fill="#10131A")

    def _placer_bas_droite(self) -> None:
        self.update_idletasks()
        x = self.winfo_screenwidth() - 460
        y = self.winfo_screenheight() - 190
        self.geometry(f"+{x}+{y}")

    def _debut_glisser(self, event) -> None:
        self._origine = (event.x, event.y)

    def _glisser(self, event) -> None:
        if getattr(self, "_origine", None):
            dx = event.x - self._origine[0]
            dy = event.y - self._origine[1]
            self.geometry(f"+{self.winfo_x() + dx}+{self.winfo_y() + dy}")

    # ---------------- chargement (threads) ----------------
    def _charger(self) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]
            modele = os.environ.get("AURA_VOCAL_MODELE", "small")
            self._whisper = WhisperModel(
                modele, device="cpu", compute_type="int8")
            self._file.put(("etape", "whisper OK (small, cpu)"))
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", "whisper: " + str(e)[:120]))
            return
        try:
            from aura.orchestrateur import Aura1B
            ia = Aura1B()
            self._file.put(("pret", ia))
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", "aura: " + str(e)[:120]))

    # ---------------- enregistrement + transcription ----------------
    def _toggle_global(self, _e=None) -> None:
        if self._ecoute:
            self._stop_ecoute()
        else:
            self._start_ecoute()

    def _start_ecoute(self) -> None:
        if self._ecoute or self._occupe or self._ia is None:
            if self._ia is None:
                self._bulle.config(text="Cerveau encore en chargement...")
            return
        self._ecoute = True
        self._dessine_rond(_ROUGE)
        self._lab_etat.config(text="j'ecoute... (relache pour envoyer)")
        self._file.put(("ecoute", None))

    def _stop_ecoute(self) -> None:
        if not self._ecoute:
            return
        self._ecoute = False
        self._dessine_rond(_JAUNE)
        self._lab_etat.config(text="transcription...")

    def _debut_parler(self, _e=None) -> None:
        self._start_ecoute()

    def _fin_parler(self, _e=None) -> None:
        if self._ecoute:
            self._stop_ecoute()

    # ---------------- boucle d'evenements ----------------
    def _pomper(self) -> None:
        try:
            while True:
                msg = self._file.get_nowait()
                genre = msg[0]
                if genre == "etape":
                    self._lab_etat.config(text=str(msg[1]))
                elif genre == "pret":
                    self._ia = msg[1]
                    self._lab_etat.config(text="pret - F9 ou maintiens le rond")
                    self._dessine_rond(_VERT)
                elif genre == "ecoute":
                    threading.Thread(target=self._enregistrer,
                                     daemon=True).start()
                elif genre == "question":
                    self._occupe = True
                    self._bulle.config(text="tu : " + str(msg[1])[:260])
                    self._lab_etat.config(text="Aura reflechit...")
                    threading.Thread(target=self._repondre,
                                     args=(msg[1],), daemon=True).start()
                elif genre == "copie":
                    try:
                        self.clipboard_clear()
                        self.clipboard_append(str(msg[1]))
                    except Exception:
                        pass
                elif genre == "reponse":
                    self._bulle.config(text="Aura : " + str(msg[1])[:260])
                    self._lab_etat.config(text="pret - F9 ou maintiens le rond")
                    self._dessine_rond(_VERT)
                    self._occupe = False
                elif genre == "erreur":
                    self._bulle.config(text="[erreur] " + str(msg[1])[:200])
                    self._lab_etat.config(text="pret - F9 ou maintiens le rond")
                    self._dessine_rond(_VERT)
                    self._occupe = False
        except queue.Empty:
            pass
        self.after(100, self._pomper)

    # ---------------- audio ----------------
    def _enregistrer(self) -> None:
        import numpy as np
        import sounddevice as sd  # type: ignore[import-untyped]
        fs = 16000
        blocs: list = []
        self._file.put(("etape", "j'ecoute... parle maintenant"))
        try:
            with sd.InputStream(samplerate=fs, channels=1, dtype="float32",
                                blocksize=800) as flux:
                while self._ecoute:
                    donnees, _ = flux.read(1600)
                    blocs.append(donnees.copy())
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", "micro: " + str(e)[:100]))
            return
        if not blocs:
            self._file.put(("erreur", "aucun audio capture"))
            return
        audio = np.concatenate(blocs).reshape(-1)
        if audio.size < fs // 2:
            self._file.put(("erreur", "trop court (moins de 0,5 s)"))
            return
        self._file.put(("etape", "transcription..."))
        try:
            segments, _ = self._whisper.transcribe(
                audio, language="fr", beam_size=1, vad_filter=True)
            texte = " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", "whisper: " + str(e)[:100]))
            return
        if not texte:
            self._file.put(("erreur", "rien entendu"))
            return
        self._file.put(("question", texte))

    # ---------------- reponse Aura ----------------
    def _repondre(self, question: str) -> None:
        try:
            t0 = time.time()
            r = self._ia.executer_detaille(question)
            reponse = str(r.get("reponse", "")).strip() or "..."
            dt = round(time.time() - t0, 1)
            self._file.put(("reponse", reponse))
            self._file.put(("copie", reponse))
            if self.avec_voix:
                self._dire(reponse)
            self._file.put(("etape", "repondu en " + str(dt) + " s"))
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", str(e)[:160]))

    def _dire(self, texte: str) -> None:
        try:
            import pyttsx3  # type: ignore[import-untyped]
            if self._tts is None:
                self._tts = pyttsx3.init()
                voix = [v for v in self._tts.getProperty("voices")
                        if "french" in v.name.lower()
                        or "fr" in (v.id or "").lower()]
                if voix:
                    self._tts.setProperty("voice", voix[0].id)
                self._tts.setProperty("rate", 170)
            self._tts.say(texte[:600])
            self._tts.runAndWait()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(prog="aura_vocal")
    ap.add_argument("--sans-voix", action="store_true",
                    help="desactive la synthese vocale")
    ap.add_argument("--smoke", action="store_true",
                    help="ferme seul apres 4 s (test)")
    args = ap.parse_args()
    Pastille(avec_voix=not args.sans_voix, smoke=args.smoke).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
