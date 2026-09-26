"""Aura-Vocal : l'encoche (notch) d'un MacBook, mais vivante.

Un petit rectangle noir aux coins inferieurs arrondis, colle au bord
superieur au centre de l'ecran, comme la notch d'un MacBook Pro :
  - au centre : la camera (capteur visible), le voyant d'activite
    (il s'allume quand Aura ecoute) et le capteur de luminosite ;
  - a gauche : le statut d'Aura (comme un menu de la barre macOS) ;
  - a droite : l'heure (comme la barre de menus).
Maintiens l'encoche pour parler, relache : ce que tu dis s'ecrit dans
la bulle sous l'encoche, Aura repond, dit la reponse et la copie.
Double-clic sur l'encoche : fermer. F9 : alternative clavier.
Transcription faster-whisper 100 % locale ; moteur Aura-1B.

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
from datetime import datetime
from typing import Any

_NOIR = "#000000"
_BULLE = "#141414"
_BLANC = "#FFFFFF"
_GRIS = "#8E8E93"
_BLEU = "#4C8DFF"
_ROUGE = "#FF5F57"
_JAUNE = "#FEBC2E"
_VERT = "#28C840"
_LED_OFF = "#1A1A1A"
_CAM = "#0D0D0D"
_MAGIQUE = "#ABC012"          # rendu 100 % transparent
_L = 560                      # largeur de l'encoche
_H = 36                       # hauteur
_NL = chr(10)


def _coins_bas(canvas: tk.Canvas, l: int, h: int, r: float,
               fill: str) -> None:
    """Rectangle plein en haut, coins INFERIEURS arrondis (notch)."""
    pts: list[float] = [0.0, 0.0, float(l), 0.0, float(l), h - r]
    for i in range(9):
        a = math.radians(90 * i / 8)
        pts.append(l - r + r * math.sin(a))
        pts.append(h - r + r * math.cos(a))
    for i in range(9):
        a = math.radians(90 * i / 8)
        pts.append(r - r * math.sin(a))
        pts.append(h - r + r * math.cos(a))
    pts.append(0.0)
    pts.append(h - r)
    canvas.create_polygon(pts, fill=fill, outline="", smooth=True)


class NotchVocale(tk.Tk):
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
        self.after(1000, self._horloge)
        self.bind("<F9>", lambda _e: self._toggle())
        if smoke:
            self.after(4000, self.destroy)

    # ---------------- construction de l'encoche ----------------
    def _construire(self) -> None:
        self._bulle = tk.Label(
            self, text="", bg=_BULLE, fg=_BLANC, font=("Segoe UI", 10),
            justify="left", anchor="w", wraplength=_L - 40, padx=14,
            pady=10)

        encoche = tk.Frame(self, bg=_MAGIQUE)
        encoche.pack(side="top")
        self._cv = tk.Canvas(encoche, width=_L, height=_H, bg=_MAGIQUE,
                             highlightthickness=0, cursor="hand2")
        self._cv.pack()
        _coins_bas(self._cv, _L, _H, 12, _NOIR)

        # --- a gauche : le menu Aura (statut, texte blanc) ---
        self._txt = self._cv.create_text(
            16, _H // 2, text="Aura   chargement...", fill=_BLANC,
            font=("Segoe UI", 9, "bold"), anchor="w")

        # --- au centre : camera + voyant + capteur (comme la notch) ---
        cx = _L // 2
        self._cv.create_oval(cx - 9, 9, cx + 9, 27, fill=_CAM,
                             outline="#222222")          # lentille camera
        self._cv.create_oval(cx - 4, 14, cx + 2, 20,
                             fill="#151B2E", outline="")  # reflet capteur
        self._led_id = self._cv.create_oval(cx + 14, 15, cx + 20, 21,
                                         fill=_LED_OFF, outline="")  # voyant
        self._cv.create_oval(cx + 26, 16, cx + 30, 20, fill=_LED_OFF,
                             outline="")                  # capteur lum.

        # --- a droite : l'heure (comme la barre de menus macOS) ---
        self._heure = self._cv.create_text(
            _L - 14, _H // 2, text="", fill=_BLANC,
            font=("Segoe UI", 9), anchor="e")

        # zone cliquable : tout le canvas
        self._cv.bind("<ButtonPress-1>", self._presser)
        self._cv.bind("<ButtonRelease-1>", self._relacher)
        self._cv.bind("<B1-Motion>", self._glisser)
        self._cv.bind("<Double-Button-1>", lambda _e: self.destroy())
        self._placer()

    def _horloge(self) -> None:
        self._cv.itemconfig(self._heure,
                            text=datetime.now().strftime("%a %H:%M"))
        self.after(20000, self._horloge)

    def _placer(self) -> None:
        self.update_idletasks()
        x = max(0, (self.winfo_screenwidth() - _L) // 2)
        self.geometry(f"+{x}+0")

    # ---------------- interactions ----------------
    def _presser(self, event) -> None:
        self._clic = (event.x, event.y)
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
                self._bulle_texte("chargement en cours...")
                return
            self._ecoute = True
            self._led(_ROUGE)
            self._statut("Aura   j'ecoute - relache pour envoyer")
            self._bulle_texte("... parle")
            self._file.put(("go", None))
        elif not activer and self._ecoute:
            self._ecoute = False
            self._led(_JAUNE)
            self._statut("Aura   transcription...")

    # ---------------- affichage ----------------
    def _bulle_texte(self, texte: str) -> None:
        if not self._bulle.winfo_ismapped():
            self._bulle.pack(side="top", fill="x")
        self._bulle.config(text=texte)

    def _statut(self, texte: str) -> None:
        self._cv.itemconfig(self._txt, text=texte)

    def _led(self, couleur: str) -> None:
        self._cv.itemconfig(self._led_id, fill=couleur)

    # ---------------- chargement + micro ----------------
    def _charger(self) -> None:
        self._demute_si_besoin()
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]
            modele = os.environ.get("AURA_VOCAL_MODELE", "small")
            self._whisper = WhisperModel(modele, device="cpu",
                                         compute_type="int8")
            self._file.put(("etape", "voix OK"))
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
                self._file.put(("micro", "AUCUN micro actif : connecte ton "
                                "casque JBL ou active le micro Realtek"))
                return
            dev = coll.Item(0)
            ptr = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            vol = cast(ptr, POINTER(IAudioEndpointVolume))
            if vol.GetMute():
                vol.SetMute(0, None)
                self._file.put(("micro", "micro etait MUET : je viens de "
                                "le demute (volume "
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
        self._file.put(("etape", "Aura   j'ecoute..."))
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
            self._file.put(("erreur", "silence capte : ton micro integre "
                            "ne transmet rien - connecte le casque JBL "
                            "(pic=" + str(round(pic, 4)) + ")"))
            return
        if pic < 0.05:
            audio = audio * (0.3 / pic)   # gain auto si micro faible
        self._file.put(("etape", "Aura   transcription..."))
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
            self._file.put(("erreur", "rien compris (pic="
                            + str(round(pic, 3)) + ")"))
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
                    self._statut(str(msg[1]))
                elif g == "niveau":
                    b = min(8, int(float(msg[1]) * 300))
                    self._statut("Aura   niveau " + "|" * b
                                 + (" faible !" if b < 2 else ""))
                elif g == "micro":
                    self._bulle_texte("[micro] " + str(msg[1]))
                elif g == "pret":
                    self._ia = msg[1]
                    self._led(_VERT)
                    self._statut("Aura   pret - maintiens l'encoche ou F9")
                elif g == "go":
                    threading.Thread(target=self._enregistrer,
                                     daemon=True).start()
                elif g == "question":
                    self._occupe = True
                    self._led(_JAUNE)
                    self._bulle_texte("tu : " + str(msg[1]))
                    self._statut("Aura   reflechit...")
                    threading.Thread(target=self._repondre,
                                     args=(msg[1],), daemon=True).start()
                elif g == "reponse":
                    q, rep, dt = str(msg[1]), str(msg[2]), msg[3]
                    self._bulle_texte("tu : " + q + _NL + "Aura : " + rep)
                    self._statut("Aura   pret - " + str(dt)
                                 + " s - copie (Ctrl+V)")
                    self._led(_VERT)
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
                    self._bulle_texte("[probleme] " + str(msg[1]))
                    self._statut("Aura   pret - reessaie")
                    self._led(_VERT)
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
                    help="ferme seul apres 4 s")
    args = ap.parse_args()
    NotchVocale(avec_voix=not args.sans_voix, smoke=args.smoke).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
