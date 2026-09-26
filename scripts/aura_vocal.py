"""Aura-Vocal : mini-barre noire en haut, deux gouttes d'eau (style TypeFlux).

Une barre fine tout-en-noir en haut de l'ecran : maintiens les gouttes
pour parler, relache -> ce que tu dis s'ecrit DANS LA BULLE AU-DESSUS,
puis Aura repond dans la meme bulle, dit a voix haute et copie au
presse-papiers. Transcription faster-whisper 100 % locale.

Usage : uv run python scripts/aura_vocal.py [--sans-voix] [--smoke]
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
import tkinter as tk
from typing import Any

_NOIR = "#0B0B0B"
_NOIR2 = "#141414"
_BLANC = "#F2F2F2"
_GRIS = "#8A8A8A"
_BLEU = "#6FB7FF"
_ROUGE = "#FF5F57"
_JAUNE = "#FEBC2E"
_VERT = "#28C840"
_NL = chr(10)


def _goutte(canvas: tk.Canvas, ox: float, oy: float, echelle: float,
            couleur: str) -> None:
    """Dessine une goutte d'eau (pointe en haut, ronde en bas)."""
    pts: list[float] = []
    import math
    for i in range(40):
        a = 2 * math.pi * i / 40
        # goutte : cercle decale vers le bas + pointe haute
        x = ox + echelle * 0.78 * math.sin(a)
        y = oy + echelle * 0.85 * math.cos(a) - (echelle * 0.55
                                                 * max(0.0, math.cos(a)))
        pts.extend((x, y))
    canvas.create_polygon(pts, fill=couleur, outline="", smooth=True)


class BarreVocale(tk.Tk):
    def __init__(self, avec_voix: bool = True, smoke: bool = False) -> None:
        super().__init__()
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(bg=_NOIR)
        self.avec_voix = avec_voix
        self._file: "queue.Queue" = queue.Queue()
        self._ia: Any = None
        self._whisper: Any = None
        self._tts: Any = None
        self._ecoute = False
        self._occupe = False
        self._origine: "tuple[int, int] | None" = None

        self._construire()
        threading.Thread(target=self._charger, daemon=True).start()
        self.after(100, self._pomper)
        self.bind("<F9>", lambda _e: self._toggle())
        if smoke:
            self.after(4000, self.destroy)

    # ---------------- UI ----------------
    def _construire(self) -> None:
        # bulle (au-dessus de la barre) : transcription + reponse
        self._bulle = tk.Label(
            self, text="", bg=_NOIR2, fg=_BLANC, font=("Segoe UI", 10),
            justify="left", anchor="w", wraplength=430,
            padx=12, pady=8)
        # barre : deux gouttes + statut
        barre = tk.Frame(self, bg=_NOIR)
        barre.pack(side="bottom")
        self._canvas = tk.Canvas(barre, width=74, height=34, bg=_NOIR,
                                 highlightthickness=0, cursor="hand2")
        self._canvas.pack(side="left", padx=6, pady=4)
        self._peindre(_BLEU)
        self._canvas.bind("<ButtonPress-1>", self._debut_presser)
        self._canvas.bind("<ButtonRelease-1>", self._fin_presser)
        self._canvas.bind("<B1-Motion>", self._glisser_si_moved)
        self._canvas.bind("<Double-Button-1>", lambda _e: self.destroy())
        self._statut = tk.Label(barre, text="chargement du cerveau...",
                                bg=_NOIR, fg=_GRIS,
                                font=("Segoe UI", 9))
        self._statut.pack(side="left", padx=(0, 10))
        self._placer_haut()

    def _peindre(self, couleur: str) -> None:
        c = self._canvas
        c.delete("all")
        _goutte(c, 20, 19, 13, couleur)
        _goutte(c, 48, 19, 13, couleur)

    def _placer_haut(self) -> None:
        self.update_idletasks()
        l = self.winfo_reqwidth()
        x = max(0, (self.winfo_screenwidth() - l) // 2)
        self._y_barre = 8
        self.geometry(f"+{x}+{self._y_barre}")

    def _recaler(self) -> None:
        """La barre reste colle en haut : la bulle pousse vers le bas."""
        pass

    def _debut_presser(self, event) -> None:
        self._clic: tuple[float, float] | None = (event.x, event.y)
        self._moved = False
        self.after(180, self._si_maintenu)

    def _si_maintenu(self) -> None:
        if getattr(self, "_clic", None) and not self._moved:
            self._toggle(on=True)

    def _glisser_si_moved(self, event) -> None:
        ox, oy = self._clic or (event.x, event.y)
        if abs(event.x - ox) + abs(event.y - oy) > 8:
            self._moved = True
            dx = event.x - ox
            dy = event.y - oy
            self.geometry(f"+{self.winfo_x() + dx}+{self.winfo_y() + dy}")

    def _fin_presser(self, _e=None) -> None:
        if getattr(self, "_moved", False):
            self._clic = None
            return
        self._toggle(on=False)

    # ---------------- etats ----------------
    def _toggle(self, on: "bool | None" = None) -> None:
        activer = (not self._ecoute) if on is None else on
        if activer and not self._ecoute and not self._occupe:
            if self._ia is None or self._whisper is None:
                self._bulle.config(text="chargement en cours...")
                return
            self._ecoute = True
            self._peindre(_ROUGE)
            self._statut.config(text="j'ecoute...", fg=_BLANC)
            self._bulle.config(text="... parle (relache pour envoyer)")
            self._file.put(("go", None))
        elif not activer and self._ecoute:
            self._ecoute = False
            self._peindre(_JAUNE)
            self._statut.config(text="transcription...", fg=_GRIS)

    # ---------------- chargement ----------------
    def _charger(self) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]
            modele = os.environ.get("AURA_VOCAL_MODELE", "small")
            self._whisper = WhisperModel(modele, device="cpu",
                                         compute_type="int8")
            self._file.put(("etape", "voix OK (" + modele + ")"))
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", "whisper: " + str(e)[:120]))
            return
        try:
            from aura.orchestrateur import Aura1B
            ia = Aura1B()
            self._file.put(("pret", ia))
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", "aura: " + str(e)[:120]))

    # ---------------- audio : RSB check + transcription ----------------
    def _enregistrer(self) -> None:
        import numpy as np
        import sounddevice as sd  # type: ignore[import-untyped]
        fs = 16000
        blocs: list = []
        self._file.put(("etape", "j'ecoute... parle maintenant"))
        try:
            with sd.InputStream(samplerate=fs, channels=1, dtype="float32",
                                blocksize=1600) as flux:
                while self._ecoute:
                    donnees, _ = flux.read(1600)
                    blocs.append(donnees.copy())
                    niv = float(np.abs(donnees).mean())
                    if len(blocs) % 10 == 0:
                        self._file.put(("niveau", niv))
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", "micro: " + str(e)[:100]))
            return
        if not blocs:
            self._file.put(("erreur", "aucun audio capture"))
            return
        audio = np.concatenate(blocs).reshape(-1).astype("float32")
        duree = audio.size / fs
        pic = float(np.abs(audio).max())
        if duree < 0.6:
            self._file.put(("erreur", "trop court (" + str(round(duree, 1))
                            + " s)"))
            return
        if pic < 0.004:
            nom_micro = ""
            try:
                nom_micro = str(sd.query_devices(kind="input")["name"])[:40]
            except Exception:
                pass
            self._file.put(("erreur", "silence capte (micro " + nom_micro
                            + " muet ? niveau max=" + str(round(pic, 4))
                            + ")"))
            return
        if pic < 0.05:
            audio = audio * (0.3 / pic)   # micro faible : on amplifie
        self._file.put(("etape", "transcription " + str(round(duree, 1))
                        + " s (pic " + str(round(pic, 3)) + ")..."))
        try:
            segments, _ = self._whisper.transcribe(
                audio, language="fr", beam_size=1,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500,
                                "speech_pad_ms": 200})
            texte = " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", "whisper: " + str(e)[:100]))
            return
        if not texte:
            self._file.put(("erreur", "parle plus fort ou plus longtemps "
                            "(pic=" + str(round(pic, 3)) + ")"))
            return
        self._file.put(("question", texte))

    # ---------------- reponse ----------------
    def _repondre(self, question: str) -> None:
        try:
            t0 = time.time()
            r = self._ia.executer_detaille(question)
            reponse = str(r.get("reponse", "")).strip() or "..."
            dt = round(time.time() - t0, 1)
            self._file.put(("reponse", question, reponse, dt))
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
            self._tts.say(texte[:500])
            self._tts.runAndWait()
        except Exception:
            pass

    # ---------------- pompe d'evenements ----------------
    def _pomper(self) -> None:
        try:
            while True:
                msg = self._file.get_nowait()
                g = msg[0]
                if g == "etape":
                    self._statut.config(text=str(msg[1]), fg=_GRIS)
                elif g == "niveau":
                    barre = min(10, int(float(msg[1]) * 400))
                    self._statut.config(text="niveau " + ("|" * barre)
                                        + ("faible !" if barre < 2 else ""),
                                        fg=_BLEU)
                elif g == "pret":
                    self._ia = msg[1]
                    self._peindre(_VERT)
                    self._statut.config(
                        text="pret - maintiens les gouttes ou F9",
                        fg=_GRIS)
                elif g == "go":
                    threading.Thread(target=self._enregistrer,
                                     daemon=True).start()
                elif g == "question":
                    self._occupe = True
                    self._peindre(_BLEU)
                    self._statut.config(text="Aura reflechit...", fg=_BLANC)
                    self._bulle.config(text="tu : " + str(msg[1]))
                    threading.Thread(target=self._repondre,
                                     args=(msg[1],), daemon=True).start()
                elif g == "reponse":
                    q, rep, dt = msg[1], str(msg[2]), msg[3]
                    self._bulle.config(text="tu : " + q + _NL + "Aura : "
                                       + rep)
                    self._statut.config(
                        text="pret - " + str(dt) + " s - copie (Ctrl+V)",
                        fg=_GRIS)
                    self._peindre(_VERT)
                    self._occupe = False
                    try:
                        self.clipboard_clear()
                        self.clipboard_append(rep)
                    except Exception:
                        pass
                    if self.avec_voix:
                        threading.Thread(target=self._dire, args=(rep,),
                                         daemon=True).start()
                elif g == "erreur":
                    self._bulle.config(text="[probleme] " + str(msg[1]))
                    self._statut.config(text="pret - reessaie", fg=_GRIS)
                    self._peindre(_VERT)
                    self._occupe = False
                    self._ecoute = False
        except queue.Empty:
            pass
        self.after(100, self._pomper)


def main() -> int:
    ap = argparse.ArgumentParser(prog="aura_vocal")
    ap.add_argument("--sans-voix", action="store_true",
                    help="desactive la synthese vocale")
    ap.add_argument("--smoke", action="store_true",
                    help="ferme seul apres 4 s (test)")
    args = ap.parse_args()
    BarreVocale(avec_voix=not args.sans_voix, smoke=args.smoke).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
