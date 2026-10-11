"""Petit classifieur MLP, code DE ZERO en numpy (aucune dependance sklearn).

Pourquoi ce module existe
-------------------------
Mesure du 8 oct. 2026 (`scripts/experiment_routeur.py`) : sur les 150 exemples
reels du routeur d'RontoGPT (vecteurs MiniLM 384-d, classes math/web/general), un
petit MLP a UNE couche cache bat le classifieur lineaire en place, en
validation croisee 10 plis x 3 graines :

    LogisticRegression 0,878 +/- 0,008   (regime actuel < 300 exemples)
    LinearSVC          0,864 +/- 0,014   (regime actuel >= 300 exemples)
    MLP ReLU           0,900 +/- 0,011   <- retenu (+2,2 pts vs LogReg)
    MLP GLU            0,889 +/- 0,003

Note honnete : le labo d'architectures avait trouve la porte GLU gagnante a
~70 000 parametres ; ici, a 150 exemples, GLU perd legerement contre ReLU
( moins de donnees = moins de place pour la porte ). On retient donc ReLU —
le gagnant se decide par la MESURE, pas par la preference.

Contrat
-------
`ClassifieurMLP` : `fit(X, y)` puis `probas1(x) -> softmax`. Entrainement
numerique pur (numpy), deterministe pour une graine donnee, serialisable
(`etat()` / `depuis_etat()`). Aucune dependance d'entrainement (sklearn n'est
plus requis pour ce classifieur).
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["ClassifieurMLP"]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class ClassifieurMLP:
    """MLP 1 couche cachee (ReLU ou GLU) + softmax, Adam, numpy."""

    def __init__(self, D: int, K: int, H: int = 48, glu: bool = False,
                 graine: int = 0) -> None:
        self.D, self.K, self.H, self.glu = D, K, H, glu
        rng = np.random.default_rng(graine)
        self.W1 = rng.normal(0, (1 / D) ** .5, (D, H))
        self.b1 = np.zeros(H)
        self.W2 = rng.normal(0, (1 / H) ** .5, (H, K))
        self.b2 = np.zeros(K)
        self.Wg = rng.normal(0, (1 / D) ** .5, (D, H)) if glu else None
        self.bg = np.zeros(H) if glu else None
        self.mu = np.zeros(D)
        self.sd = np.ones(D)
        self.classes_: list[str] = []

    # -- entrainement ----------------------------------------------------

    def _norm(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def _avant(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xn = self._norm(X)
        pre = Xn @ self.W1 + self.b1
        if self.glu:
            assert self.Wg is not None and self.bg is not None
            g = 1.0 / (1.0 + np.exp(-np.clip(Xn @ self.Wg + self.bg, -30, 30)))
            h = pre * g
        else:
            h = np.maximum(pre, 0.0)
        return h, _softmax(h @ self.W2 + self.b2)

    def fit(self, X: np.ndarray, y: list[str] | np.ndarray, *,
            epochs: int = 400, lr: float = 2e-3, wd: float = 1e-4,
            lot: int = 32, graine: int = 0) -> "ClassifieurMLP":
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = sorted({str(c) for c in y})
        idx = {c: i for i, c in enumerate(self.classes_)}
        yy = np.array([idx[str(c)] for c in y])
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        rng = np.random.default_rng(graine)
        params: list[np.ndarray] = [self.W1, self.b1, self.W2, self.b2]
        if self.glu:
            assert self.Wg is not None and self.bg is not None
            params += [self.Wg, self.bg]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        for _ in range(epochs):
            perm = rng.permutation(len(X))
            for i in range(0, len(perm), lot):
                j = perm[i:i + lot]
                Xb, yb = X[j], yy[j]
                grads = self._graphes(Xb, yb)
                gs = [grads["W1"], grads["b1"], grads["W2"], grads["b2"]]
                if self.glu:
                    gs += [grads["Wg"], grads["bg"]]
                t += 1
                for k, p in enumerate(params):
                    g = gs[k] + (wd * p if k in (0, 2) else 0.0)
                    m[k] = .9 * m[k] + .1 * g
                    v[k] = .999 * v[k] + .001 * g * g
                    p -= lr * (m[k] / (1 - .9 ** t)) / (
                        np.sqrt(v[k] / (1 - .999 ** t)) + 1e-8)
        return self

    def _graphes(self, X: np.ndarray, yy: np.ndarray) -> dict[str, np.ndarray]:
        """Gradient du pas courant (utile aussi au test de differences finies)."""
        Xn = self._norm(X)
        pre = Xn @ self.W1 + self.b1
        if self.glu:
            assert self.Wg is not None and self.bg is not None
            gi = Xn @ self.Wg + self.bg
            g = 1.0 / (1.0 + np.exp(-np.clip(gi, -30, 30)))
            h = pre * g
        else:
            h = np.maximum(pre, 0.0)
        p = _softmax(h @ self.W2 + self.b2)
        B = len(X)
        dl = p.copy()
        dl[np.arange(B), yy] -= 1.0
        dl /= B
        grads: dict[str, np.ndarray] = {}
        grads["W2"] = h.T @ dl
        grads["b2"] = dl.sum(0)
        dh = dl @ self.W2.T
        if self.glu:
            grads["W1"] = Xn.T @ (dh * g)
            grads["b1"] = (dh * g).sum(0)
            dg = dh * pre * g * (1 - g)
            grads["Wg"] = Xn.T @ dg
            grads["bg"] = dg.sum(0)
        else:
            grads["W1"] = Xn.T @ (dh * (pre > 0))
            grads["b1"] = dh.sum(0)
        return grads

    # -- inference -------------------------------------------------------

    def probas1(self, x: np.ndarray) -> np.ndarray:
        _, p = self._avant(np.asarray(x, dtype=np.float64).reshape(1, -1))
        return p[0]

    # -- serialisation ---------------------------------------------------

    def etat(self) -> dict[str, Any]:
        return {
            "D": self.D, "K": self.K, "H": self.H, "glu": self.glu,
            "classes": self.classes_, "mu": self.mu, "sd": self.sd,
            "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
            "Wg": self.Wg, "bg": self.bg,
        }

    @classmethod
    def depuis_etat(cls, etat: dict[str, Any]) -> "ClassifieurMLP":
        obj = cls(int(etat["D"]), int(etat["K"]), int(etat["H"]),
                  bool(etat["glu"]), graine=0)
        obj.classes_ = list(etat["classes"])
        obj.mu, obj.sd = etat["mu"], etat["sd"]
        obj.W1, obj.b1, obj.W2, obj.b2 = (etat["W1"], etat["b1"],
                                          etat["W2"], etat["b2"])
        obj.Wg, obj.bg = etat["Wg"], etat["bg"]
        return obj
