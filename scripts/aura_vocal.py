"""Aura-Vocal : mini-barre noire arrondie en haut, boutons ronds style Apple.

Deux cercles parfaits (comme les feux tricolores macOS) : le bleu =
maintiens pour parler (rouge pendant l'ecoute), le rouge = fermer.
Bords arrondis (fenetre a fond transparent), texte blanc, barre collee
au bord superieur de l'ecran. Transcription faster-whisper 100 % locale,
reponse par Aura, dite a voix haute et copiee.

Usage : uv run python scripts/aura_vocal.py [--sans-voix] [--smoke]
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import threading
import time
import tkinter as tk
from typing import Any

_NOIR = "#101010"
_NOIR2 = "#181818"
_BLANC = "#FFFFFF"
_GRIS = "#9A9A9A"
_BLEU = "#4C8DFF"
_ROUGE = "#FF5F57"
_JAUNE = "#FEBC2E"
_VERT = "#28C840"
_MAGIQUE = "#ABC012"          # couleur rendue 100 % transparente
_NL = chr(10)


def _rounded(canvas: tk.Canvas, x0: float, y0: float, x1: float, y1: float,
             r: float, fill: str) -> None:
    """Rectangle aux coins arrondis (polygone lisse)."""
    pts: list[float] = []
    for cx, cy, a0 in ((x1 - r, y0 + r, 270), (x1 - r, y1 - r, 0),
                       (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for i in range(9):
            a = math.radians(a0 + 10 * i)
            pts.extend((cx + r * math.cos(a), cy + r * math.sin(a)))
    canvas.create_polygon(pts, fill=fill, outline="", smooth=True)


def _cercle(canvas: tk.Canvas, cx: float, cy: float, r: float,
            fill: str, glyphe: str = "") -> None:
    canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill,
                       outline="#3A3A3A")
    if glyphe:
        canvas.create_text(cx, cy + 1, text=glyphe, fill="#5A1010",
                           font=("Segoe UI", 10, "bold"))


class BarreVocale(tk.Tk):
    def __init__(self, avec_voix: bool = True, smoke: bool = False) -> None:
        super().__init__()
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(bg=_MAGIQUE)
        self.attributes("-transparentcolor", _MAGIQUE)
        self.avec_voix = avec_voix
        self._file: "queue.Queue" = queue.Queue()
        self._ia: Any = None
        self._whisper: Any = None
        self._tts: Any = None
        self._ecoute = False
        self._occupe = False
        self._clic: "tuple[float, float] | None" = None
        self._moved = False

        self._construire()
        threading.Thread(target=self._charger, daemon=True).start()
        self.after(100, self._pomper)
        self.bind("<F9>", lambda _e: self._toggle())
        if smoke:
            self.after(4000, self.destroy)

    # ---------------- construction UI ----------------
    def _construire(self) -> None:
        # bulle (sous la barre, barre collee au bord haut)
        self._bulle = tk.Label(
            self, text="", bg=_NOIR2, fg=_BLANC, font=("Segoe UI", 10),
            justify="left", anchor="w", wraplength=430, padx=14, pady=10)
        self._bulle.pack_forget()

        barre = tk.Frame(self, bg=_MAGIQUE)
        barre.pack(side="top")
        self._canvas = tk.Canvas(barre, width=250, height=44, bg=_MAGIQUE,
                                 highlightthickness=0, cursor="hand2")
        self._canvas.pack()
        _rounded(self._canvas, 1, 1, 249, 43, 14, _NOIR)
        self._paint_cercles(_BLEU)
        self._txt = self._canvas.create_text(
            150, 22, text="chargement du cerveau...", fill=_GRIS,
            font=("Segoe UI", 9), anchor="w", width=150)
        self._canvas.bind("<ButtonPress-1>", self._presser)
        self._canvas.bind("<ButtonRelease-1>", self._relacher)
        self._canvas.bind("<B1-Motion>", self._glisser)
        self._placer_haut()

    def _paint_cercles(self, couleur_mic: str) -> None:
        c = self._canvas
        c.delete("mic", "fermer")
        if couleur_mic == _ROUGE:
            _cercle(c, 26, 22, 13, _ROUGE, "")
        else:
            _cercle(c, 26, 22, 13, couleur_mic, "")
        _cercle(c, 60, 22, 13, _ROUGE, chr(215))
        c.addtag_withtag("mic", "all")

    # ---------------- interactions ----------------
    def _presser(self, event) -> None:
        x, y = event.x, event.y
        if (x - 60) ** 2 + (y - 22) ** 2 <= 169:      # cercle rouge : fermer
            self.destroy()
            return
        self._clic = (x, y)
        self._moved = False
        self.after(160, self._si_maintenu)

    def _si_maintenu(self) -> None:
        if self._clic is not None and not self._moved:
            self._toggle(on=True)

    def _glisser(self, event) -> None:
        if self._clic is None:
            return
        ox, oy = self._clic
        if abs(event.x - ox) + abs(event.y - oy) > 9:
            self._moved = True
            self.geometry(f"+{self.winfo_x() + event.x - ox}"
                          f"+{self.winfo_y() + event.y - oy}")

    def _relacher(self, _e=None) -> None:
        if self._moved:
            self._clic = None
            return
        self._toggle(on=False)

    def _toggle(self, on: "bool | None" = None) -> None:
        activer = (not self._ecoute) if on is None else on
        if activer and not self._ecoute and not self._occupe:
            if self._ia is None or self._whisper is None:
                self._bulle.config(text="chargement en cours...")
                return
            self._ecoute = True
            self._paint_mic(_ROUGE)
            self._ecrire("... parle (relache pour envoyer)")
            self._file.put(("go", None))
        elif not activer and self._ecoute:
            self._ecoute = False
            self._paint_mic(_JAUNE)
            self._ecrire_statut("transcription...")

    # ---------------- affichage ----------------
    def _ecrire(self, texte: str) -> None:
        if not self._bulle.winfo_ismapped():
            self._bulle.pack(side="top", fill="x")
        self._bulle.config(text=texte)

    def _ecrire_statut(self, texte: str) -> None:
        self._canvas.itemconfig(self._txt, text=texte)

    def _paint_mic(self, couleur: str) -> None:
        c = self._canvas
        c.delete("mic")
        c.create_oval(13, 9, 39, 35, fill=couleur, outline="#3A3A3A",
                      tags=("mic",))

    def _placer_haut(self) -> None:
        self.update_idletasks()
        l = self.winfo_reqwidth()
        x = max(0, (self.winfo_screenwidth() - l) // 2)
        self.geometry(f"+{x}+0")

    # ---------------- chargement ----------------
    def _charger(self) -> None:
        self._demute_si_besoin()
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

    def _demute_si_besoin(self) -> None:
        """Micro muet au niveau systeme : demute et previent (cas reel)."""
        try:
            from ctypes import cast, POINTER
            import comtypes  # type: ignore[import-untyped]
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import (  # type: ignore[import-untyped]
                IMMDeviceEnumerator, IAudioEndpointVolume)
            from pycaw.constants import (  # type: ignore[import-untyped]
                CLSID_MMDeviceEnumerator, EDataFlow, DEVICE_STATE)
            en = comtypes.CoCreateInstance(CLSID_MMDeviceEnumerator,
                                           IMMDeviceEnumerator, CLSCTX_ALL)
            coll = en.EnumAudioEndpoints(EDataFlow.eCapture.value,
                                         DEVICE_STATE.ACTIVE.value)
            if coll.GetCount() == 0:
                self._file.put(("micro", "AUCUN micro actif : branche un "
                                "casque (JBL) ou active le micro Realtek"))
                return
            dev = coll.Item(0)
            ptr = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            vol = cast(ptr, POINTER(IAudioEndpointVolume))
            if vol.GetMute():
                vol.SetMute(0, None)
                self._file.put(("micro", "ton micro etait MUET : je viens "
                                "de le demute (volume "
                                + str(round(vol.GetMasterVolumeLevelScalar()
                                            * 100)) + " %)"))
        except Exception:
            pass

    # ---------------- audio ----------------
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
                    if len(blocs) % 8 == 0:
                        niv = float(np.abs(donnees).mean())
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
            self._file.put(("erreur", "silence capte : micro muet ou mauvais "
                            "peripherique (pic=" + str(round(pic, 4)) + ")"))
            return
        if pic < 0.05:
            audio = audio * (0.3 / pic)   # gain auto si micro faible
        self._file.put(("etape", "transcription " + str(round(duree, 1))
                        + " s..."))
        try:
            segments, _ = self._whisper.transcribe(
                audio, language="fr", beam_size=1, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500,
                                "speech_pad_ms": 200})
            texte = " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", "whisper: " + str(e)[:100]))
            return
        if not texte:
            self._file.put(("erreur", "rien compris : parle plus fort "
                            "(pic=" + str(round(pic, 3)) + ")"))
            return
        self._file.put(("question", texte))

    # ---------------- reponse + voix ----------------
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
                    self._ecrire_statut(str(msg[1]))
                elif g == "niveau":
                    b = min(8, int(float(msg[1]) * 300))
                    self._ecrire_statut("niveau " + "|" * b
                                        + (" faible !" if b < 2 else ""))
                elif g == "micro":
                    self._ecrire("[micro] " + str(msg[1]))
                elif g == "pret":
                    self._ia = msg[1]
                    self._paint_mic(_BLEU)
                    self._ecrire_statut("pret - maintiens le rond bleu")
                elif g == "go":
                    threading.Thread(target=self._enregistrer,
                                     daemon=True).start()
                elif g == "question":
                    self._occupe = True
                    self._paint_mic(_JAUNE)
                    self._ecrire("tu : " + str(msg[1]))
                    self._ecrire_statut("Aura reflechit...")
                    threading.Thread(target=self._repondre,
                                     args=(msg[1],), daemon=True).start()
                elif g == "reponse":
                    q, rep, dt = str(msg[1]), str(msg[2]), msg[3]
                    self._ecrire("tu : " + q + _NL + "Aura : " + rep)
                    self._ecrire_statut("pret - " + str(dt)
                                        + " s - copie (Ctrl+V)")
                    self._paint_mic(_BLEU)
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
                    self._ecrire("[probleme] " + str(msg[1]))
                    self._ecrire_statut("pret - reessaie")
                    self._paint_mic(_BLEU)
                    self._occupe = False
                    self._ecoute = False
        except queue.Empty:
            pass
        self.after(100, self._pomper)


def main() -> int:
    ap = argparse.ArgumentParser(prog="aura_vocal")
    auto = ap.add_argument
    auto("--sans-voix", action="store_true",
         help="desactive la synthese vocale")
    auto("--smoke", action="store_true", help="ferme seul apres 4 s")
    args = ap.parse_args()
    BarreVocale(avec_voix=not args.sans_voix, smoke=args.smoke).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
