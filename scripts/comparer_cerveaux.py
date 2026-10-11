"""Quiz comparatif des cerveaux : MiniCPM5-1B vs Llama 3.2 1B (et autres).

Meme quiz, memes reglages (temperature 0.4, config auto), seuls les poids
changent. Chaque reponse est verifiee par un critere objectif (sous-chaine
attendue). Mesure la latence et le debit approximatif.

Usage :
  python scripts/comparer_cerveaux.py                 # cerveau par defaut
  AURA_GGUF=LLAMA python scripts/comparer_cerveaux.py # cerveau historique
  AURA_GGUF_CHEMIN=modeles/x.gguf python ...          # n'importe quel GGUF
"""
import os
import sys
import threading
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from aura import llama_cerveau  # noqa: E402

QUIZ = [
    # (question, sous-chaines attendues, max_tokens)
    ("Quelle est la capitale de la France ?",
     ["Paris"], 40),
    ("Quelle est la capitale de l'Australie ?",
     ["Canberra"], 40),
    ("Qui a ecrit Les Miserables ?",
     ["Hugo"], 40),
    ("Quel est le carre de 12 ?",
     ["144"], 60),
    ("Traduis en anglais : « Bonjour, comment vas-tu ? »",
     ["hello", "Hello", "how are you", "How are you"], 60),
    ("Explique en une phrase ce qu'est la photosynthese.",
     ["lumiere", "lumière", "solaire", "energie", "énergie",
      "chlorophylle", "plante"], 120),
    ("Write a Python function that returns the square of a number.",
     ["def ", "return"], 200),
]

nom = Path(llama_cerveau._CHEMIN_GGUF).name
print(f"=== Cerveau : {nom} ===\n")

# Le pic instantane pris apres le chargement sous-estime fortement le besoin
# reel. Echantillonner le RSS pendant tout le quiz donne une borne empirique
# reproductible pour ce processus (poids + contexte + Python), pas pour toute
# la machine ni pour un autre profil de contexte.
try:
    import psutil
    _processus = psutil.Process(os.getpid())
    _pic_rss = [0]
    _mesure_fin = threading.Event()

    def _mesurer_rss():
        while not _mesure_fin.is_set():
            try:
                _pic_rss[0] = max(_pic_rss[0], _processus.memory_info().rss)
            except psutil.Error:
                pass
            _mesure_fin.wait(0.05)

    _fil_rss = threading.Thread(target=_mesurer_rss, daemon=True)
    _fil_rss.start()
except ImportError:
    _mesure_fin = None
    _pic_rss = None

llm = llama_cerveau._charger()
score, total = 0, len(QUIZ)
temps_total, tokens_total = 0.0, 0

for question, attendus, max_tokens in QUIZ:
    t0 = time.time()
    reponse = llama_cerveau.generer(question, max_tokens=max_tokens)
    duree = time.time() - t0
    temps_total += duree
    tokens_total += max_tokens        # borne haute (max_tokens demandes)
    ok = any(a.lower() in reponse.lower() for a in attendus)
    score += ok
    statut = "OK " if ok else "KO "
    extrait = " ".join(reponse.split())[:80]
    print(f"[{statut}] {question[:50]:50} {duree:5.1f}s  {extrait}")

print(f"\n=== Score : {score}/{total} | temps total {temps_total:.0f}s "
      f"(~{temps_total/total:.1f}s/reponse) ===")
if _mesure_fin is not None:
    _mesure_fin.set()
    _fil_rss.join(timeout=1)
if _pic_rss is not None:
    print(f"Pic RSS processus (echantillonne a 50 ms) : "
          f"{_pic_rss[0] / (1024**2):.1f} MiB")
else:
    print("Pic RSS processus : non mesure (installer psutil)")
print(f"(cerveau : {nom})")
