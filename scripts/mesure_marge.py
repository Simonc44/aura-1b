"""Mesure la marge de confiance top-1/top-2 par token sur le GGUF d'Aura.

Jeu etiquete : chaque question a un mot-cle attendu dans la reponse.
On regarde si la marge minimale separe les reponses justes des fausses
(base pour calibrer SEUIL_MARGE dans aura/intercepteur_entropie.py).
Usage : PYTHONPATH=. python scripts/mesure_marge.py
"""
import statistics
import sys

from llama_cpp import Llama

from aura import llama_cerveau
from aura.intercepteur_entropie import marge_top2, premier_doute

# (question, mot-cle attendu dans la reponse, normalise en minuscules sans accents simples)
JEU = [
    ("Combien font 17 * 23 ?", "391"),
    ("Combien font 12 * 12 ?", "144"),
    ("Combien font 25 + 37 ?", "62"),
    ("Quelle est la capitale de l'Australie ?", "canberra"),
    ("Quelle est la capitale du Canada ?", "ottawa"),
    ("Quelle est la capitale du Japon ?", "tokyo"),
    ("Quel est le plus grand océan du monde ?", "pacifique"),
    ("Combien de continents y a-t-il sur Terre ?", "7"),
    ("Quel est le symbole chimique de l'or ?", "au"),
    ("Qui a écrit Les Misérables ?", "hugo"),
    ("Quelle est la capitale de la Nouvelle-Zélande ?", "wellington"),
    ("Qui a peint la Joconde ?", "vinci"),
    ("Quelle est la capitale de la Mongolie ?", "ulaan"),
    ("Combien de planètes compte le système solaire ?", "8"),
    ("Qui a écrit Faust ?", "goethe"),
    ("En quelle année la Bastille a-t-elle été prise ?", "1789"),
    ("Quel est le nombre premier qui suit 7 ?", "11"),
    ("Quel est le plus long fleuve d'Afrique ?", "nil"),
    ("Combien font 999 * 999 ?", "998001"),
    ("Combien font 2 puissance 20 ?", "1048576"),
    ("Quel est le 15e nombre de Fibonacci, en partant de F1 = F2 = 1 ?", "610"),
    ("Quel est le 10e nombre premier ?", "29"),
    ("Quelle est la racine carrée de 144 ?", "12"),
    ("Quel est le numéro atomique du fer ?", "26"),
    ("Combien de bits contient un octet ?", "8"),
    ("Qui a découvert la pénicilline ?", "fleming"),
    ("Quelle est la capitale de la Bolivie ?", "sucre"),
    ("Combien de jours compte une année bissextile ?", "366"),
    ("Quel est le plus petit pays du monde ?", "vatican"),
    ("Combien d'os compte le corps humain adulte ?", "206"),
]


def _juste(reponse: str, attendu: str) -> bool:
    texte = reponse.lower()
    if attendu.isdigit():
        import re
        return re.search(rf"(?<!\d){attendu}(?!\d)", texte) is not None
    return attendu in texte


def main() -> int:
    llm = Llama(model_path=llama_cerveau.chemin_gguf(), n_ctx=2048,
                n_gpu_layers=0, logits_all=True, verbose=False)
    lignes = []
    for q, attendu in JEU:
        sortie = llm.create_chat_completion(
            messages=[{"role": "user", "content": q}],
            max_tokens=60, temperature=0.0,
            logprobs=True, top_logprobs=2)
        choix = sortie["choices"][0]
        texte = choix["message"]["content"] or ""
        contenu = choix["logprobs"]["content"]
        marges = [marge_top2({c["token"]: c["logprob"] for c in t["top_logprobs"]})
                  for t in contenu]
        juste = _juste(texte, attendu)
        mini = min(marges) if marges else float("nan")
        lignes.append((juste, mini))
        print(f"[{'OK ' if juste else 'KO '}] {q!r}: min={mini:.3f} "
              f"med={statistics.median(marges):.3f} "
              f"doute<0.15={premier_doute(marges)} | {texte[:60]!r}")
    justes = [m for j, m in lignes if j]
    fausses = [m for j, m in lignes if not j]
    print(f"justes: n={len(justes)} min-moyen={statistics.mean(justes):.3f}" if justes else "justes: 0")
    print(f"fausses: n={len(fausses)} min-moyen={statistics.mean(fausses):.3f}" if fausses else "fausses: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
