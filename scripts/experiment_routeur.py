"""Le routeur flou d'RontoGPT : une porte GLU bat-elle le LinearSVC actuel ?

Contexte
--------
`aura/classificateur_flou.py` entraine un LinearSVC sur les vecteurs MiniLM
(384-d) pour aiguiller une question vers {math, web, general}. Le labo
d'architectures (`scripts/lab_architectures.py`) a trouve qu'une PORTE
multiplicative (GLU) bat un MLP ReLU — sur une tache de langage.

Avant de modifier RontoGPT, on MESURE : sur les 150 exemples reels du routeur, la
porte GLU bat-elle le modele en place, en validation croisee ? Si non, on ne
touche a rien (on ne degrade pas RontoGPT pour une intuition).

Protocole : validation croisee stratifiee 5 plis, meme standardisation, meme
budget. Aucune fuite : le SCALER est ajuste sur le train de chaque pli.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _mlp(Xtr, ytr, Xte, glu=True, H=48, epochs=400, lr=2e-3, wd=1e-4,
         lot=32, graine=0):
    """MLP 1 couche (ReLU ou GLU) + softmax, numpy + Adam, depuis zero."""
    rng = np.random.default_rng(graine)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    D = Xtr.shape[1]
    K = len(set(ytr))
    classes = sorted(set(ytr))
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[c] for c in ytr])
    W1 = rng.normal(0, (1 / D) ** .5, (D, H))
    b1 = np.zeros(H)
    W2 = rng.normal(0, (1 / H) ** .5, (H, K))
    b2 = np.zeros(K)
    params = [W1, b1, W2, b2]
    if glu:
        Wg = rng.normal(0, (1 / D) ** .5, (D, H))
        bg = np.zeros(H)
        params += [Wg, bg]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]

    def fwd(X):
        h = X @ W1 + b1
        if glu:
            g = 1 / (1 + np.exp(-np.clip(X @ Wg + bg, -30, 30)))
            h = h * g
        else:
            h = np.maximum(h, 0)
        return h, _softmax(h @ W2 + b2)

    t = 0
    for _ in range(epochs):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), lot):
            j = perm[i:i + lot]
            Xb, yb = Xtr[j], y[j]
            h, p = fwd(Xb)
            B = len(Xb)
            dl = p.copy()
            dl[np.arange(B), yb] -= 1
            dl /= B
            gW2 = h.T @ dl
            gb2 = dl.sum(0)
            dh = dl @ W2.T
            if glu:
                g = 1 / (1 + np.exp(-np.clip(Xb @ Wg + bg, -30, 30)))
                gW1 = Xb.T @ (dh * g)
                gb1 = (dh * g).sum(0)
                dg = dh * (Xb @ W1 + b1) * g * (1 - g)
                gWg = Xb.T @ dg
                gbg = dg.sum(0)
                grads = [gW1, gb1, gW2, gb2, gWg, gbg]
            else:
                gW1 = Xb.T @ (dh * (Xb @ W1 + b1 > 0))
                gb1 = dh.sum(0)
                grads = [gW1, gb1, gW2, gb2]
            t += 1
            for k in range(len(params)):
                gr = grads[k] + wd * params[k]
                m[k] = .9 * m[k] + .1 * gr
                v[k] = .999 * v[k] + .001 * gr * gr
                params[k] -= lr * (m[k] / (1 - .9 ** t)) / (np.sqrt(v[k] / (1 - .999 ** t)) + 1e-8)
    return np.array([classes[i] for i in fwd(Xte)[1].argmax(1)])


def _cv(nom, pred_fn, X, y, plis=5, graine=0):
    from sklearn.model_selection import StratifiedKFold
    accs = []
    skf = StratifiedKFold(n_splits=plis, shuffle=True, random_state=graine)
    for tr, te in skf.split(X, y):
        pred = pred_fn(X[tr], y[tr], X[te])
        accs.append(float(np.mean(pred == y[te])))
    return np.mean(accs), np.std(accs)


def main() -> int:
    from aura import embeddings, routeur
    if not embeddings.actif():
        print("MiniLM indisponible : mesure impossible ici.")
        return 2
    questions, labels = routeur.exemples_entrainement()
    y = np.array(labels)
    X = embeddings.encoder(questions)
    print(f"{len(questions)} exemples, embeddings {X.shape}, "
          f"classes {sorted(set(labels))}\n")

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    def logreg(Xtr, ytr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        c = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd, ytr)
        return c.predict((Xte - mu) / sd)

    def lsvc(Xtr, ytr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        c = LinearSVC(C=1.0, dual="auto").fit((Xtr - mu) / sd, ytr)
        return c.predict((Xte - mu) / sd)

    resultats = [
        ("LogisticRegression (regime actuel <300)", logreg),
        ("LinearSVC (regime actuel >=300)", lsvc),
        ("MLP ReLU", lambda a, b, c: _mlp(a, b, c, glu=False)),
        ("MLP GLU (gagnant du labo)", lambda a, b, c: _mlp(a, b, c, glu=True)),
    ]
    # Protocole robuste : 10 plis, moyennes sur 3 graines de decoupage.
    print("Validation croisee 10 plis x 3 graines (accuracy moyenne) :")
    scores = {}
    for nom, fn in resultats:
        accs = []
        for g in (0, 1, 2):
            moy, _ = _cv(nom, fn, X, y, plis=10, graine=g)
            accs.append(moy)
        scores[nom] = (float(np.mean(accs)), float(np.std(accs)))
        print(f"  {nom:40s} {scores[nom][0]:.3f} +/- {scores[nom][1]:.3f}")

    best_sk = max(scores["LogisticRegression (regime actuel <300)"][0],
                  scores["LinearSVC (regime actuel >=300)"][0])
    best_nn = max(scores["MLP ReLU"][0], scores["MLP GLU (gagnant du labo)"][0])
    nom_nn = max(("MLP ReLU", "MLP GLU (gagnant du labo)"),
                 key=lambda n: scores[n][0])
    print()
    if best_nn > best_sk + 0.02:
        print(f"-> meilleur reseau = {nom_nn} ({best_nn:.3f}) vs sklearn "
              f"({best_sk:.3f}) : +{100*(best_nn-best_sk):.1f} pts -> on integre.")
    else:
        print(f"-> meilleur reseau ({best_nn:.3f}) ne bat pas sklearn "
              f"({best_sk:.3f}) : on NE modifie PAS RontoGPT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
