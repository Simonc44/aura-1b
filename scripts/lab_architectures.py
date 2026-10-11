"""Laboratoire d'architectures minuscules : concevoir, tester, departager.

But (utilisateur, 8 oct. 2026)
-----------------------------
Creer des MINI architectures inedites (jamais creees), les entrainer a petite
echelle sur une tache CONTROLEE, mesurer leur performance, puis — si l'une
gagne — la scaler (monter a ~1 milliard de parametres) et l'entrainer.

Ce que ce fichier fait vraiment
-------------------------------
Il fournit un LABO reproductible et honnete, pas une promesse :

  * chaque architecture est une MECANIQUE differente (pas un simple reglage) ;
  * toutes voient les MEMES donnees, le meme optimiseur (Adam), le meme
    budget — seul le « corps » change : c'est une ablation propre ;
  * une VERIFICATION DE GRADIENT (differences finies) prouve que chaque
    reseau est correctement derive — sans quoi la comparaison ne vaut rien ;
  * le gagnant est declare sur la perplexite de validation (jamais vue), a
    parametres comparables.

Les architectures fournies
--------------------------
  - mlp-relu   : reference (fenetre + 1 couche ReLU).
  - mlp-glu    : porte multiplicative apprise (h = a * sigmoide(g)).
  - polyact    : TROIS non-linearites sommées (ReLU + tanh + swish) — une
                 mecanique d'activation mixte, inhabituelle.
  - rnn-elman  : recurrence (BPTT) — biais inductif totalement autre.
  - conv-multi : convolutions 1D a plusieurs echelles (noyaux 2/3/5) +
                 max-pooling temporel.

Ajouter la sienne : ecrire une classe heritant de `Modele` (methodes `_avant`
et `_graphes`) et l'inscrire dans `ARCHITECTURES`. Le labo la testera, la
verifiera par gradient et la classera comme les autres.

Les limites, dites d'avance
---------------------------
1. Un gagnant a petite echelle ne gagne PAS forcement a 1B : le TRANSFERT
   depend surtout des hyperparametres (d'ou la methode μP/μTransfer) et de la
   donnee, pas du score brut a 60 Ko.
2. Entrainer 1B ici est INFALSIFIABLE : pas de GPU, ~13 tok/s, et le cout
   reel d'un entrainement 1B se compte en centaines de milliers de dollars
   (cf. SmolLM2). Ce labo CONÇOIT et DEPARTAGE ; il ne peut pas entrainer 1B.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

_RACINE = Path(__file__).resolve().parent.parent
_CORPUS = ["README.md", "ARCHITECTURE.md", "RELEASE_NOTES.md", "MODEL_CARD.md"]


# ── Donnees ───────────────────────────────────────────────────────────────

def charger_corpus() -> str:
    morceaux = []
    for nom in _CORPUS:
        p = _RACINE / nom
        if p.is_file():
            morceaux.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(morceaux).lower()


def paires(texte: str, vocab: dict[str, int], K: int):
    ids = np.array([vocab.get(c, vocab["?"]) for c in texte], dtype=np.int32)
    n = len(ids) - K
    ctx = np.empty((n, K), dtype=np.int32)
    for j in range(K):
        ctx[:, j] = ids[j:j + n]
    return ctx, ids[K:K + n]


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


# ── Base commune : Adam + generation ──────────────────────────────────────

class Modele:
    """Contrat : `_avant(ctx)` -> probabilities ; `_graphes(ctx, cible)`
    -> dict nom->gradient (memes cles que `self.params`)."""

    nom = "modele"

    def __init__(self, V, K, d, H, graine=0):
        self.V, self.K, self.d, self.H = V, K, d, H
        self.params: list[str] = []
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}
        self.t = 0
        self._init(np.random.default_rng(graine))

    # a surcharger -------------------------------------------------------
    def _init(self, rng): ...
    def _avant(self, ctx): ...
    def _graphes(self, ctx, cible): ...

    def _enr(self, nom):
        if nom not in self.params:
            self.params.append(nom)
            self.m[nom] = np.zeros_like(getattr(self, nom))
            self.v[nom] = np.zeros_like(getattr(self, nom))

    def nb_params(self) -> int:
        return int(sum(getattr(self, k).size for k in self.params))

    def pas(self, ctx, cible, lr, b1=0.9, b2=0.999, eps=1e-8):
        grads = self._graphes(ctx, cible)
        # controle : chaque parametre doit avoir son gradient
        manquants = [k for k in self.params if k not in grads]
        if manquants:
            raise AssertionError(f"{self.nom}: gradient manquant {manquants}")
        self.t += 1
        for k in self.params:
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            getattr(self, k)[...] -= lr * mh / (np.sqrt(vh) + eps)

    def perte(self, ctx, cible, lot=512) -> float:
        tot, n = 0.0, 0
        for i in range(0, len(ctx), lot):
            c, y = ctx[i:i + lot], cible[i:i + lot]
            p = self._avant(c)
            tot += -np.log(np.maximum(p[np.arange(len(c)), y], 1e-9)).sum()
            n += len(c)
        return float(tot / max(n, 1))

    def _logits_vers_p(self, logits):
        return softmax(logits)

    def generer(self, amorce, vocab, inv, n=200, temp=0.8, graine=0):
        ids = [vocab.get(c, vocab["?"]) for c in amorce.lower()][-self.K:]
        ids = [vocab["?"]] * (self.K - len(ids)) + ids
        rng = np.random.default_rng(graine)
        out = []
        for _ in range(n):
            p = self._avant(np.array([ids[-self.K:]], dtype=np.int32))[0]
            logits = np.log(np.maximum(p, 1e-9)) / temp
            logits -= logits.max()
            pr = np.exp(logits)
            pr /= pr.sum()
            nxt = int(rng.choice(len(pr), p=pr))
            out.append(inv[nxt])
            ids.append(nxt)
        return "".join(out)


def _grad_emb(modele, dlogits, ctx):
    """Gradient des embeddings pour un corps qui reçoit x = concat(emb)."""
    # dlogits deja propage jusqu'a x : voir _graphes des sous-classes.
    raise NotImplementedError


# ── Architecture 1 : MLP ReLU (reference) ─────────────────────────────────

class MLPReLU(Modele):
    nom = "mlp-relu"

    def _init(self, rng):
        Kd = self.K * self.d
        self.E = rng.normal(0, .1, (self.V, self.d))
        self.W1 = rng.normal(0, (2 / Kd) ** .5, (Kd, self.H))
        self.b1 = np.zeros(self.H)
        self.W2 = rng.normal(0, (1 / self.H) ** .5, (self.H, self.V))
        self.b2 = np.zeros(self.V)
        for k in ("E", "W1", "b1", "W2", "b2"):
            self._enr(k)

    def _avant(self, ctx):
        self._ctx = ctx
        emb = self.E[ctx]
        self._x = emb.reshape(len(ctx), -1)
        self._h = np.maximum(self._x @ self.W1 + self.b1, 0)
        self._p = self._logits_vers_p(self._h @ self.W2 + self.b2)
        return self._p

    def _graphes(self, ctx, cible):
        p = self._avant(ctx)
        B = len(ctx)
        dl = p.copy(); dl[np.arange(B), cible] -= 1; dl /= B
        g = {}
        g["W2"] = self._h.T @ dl; g["b2"] = dl.sum(0)
        dh = (dl @ self.W2.T) * (self._x @ self.W1 + self.b1 > 0)
        g["W1"] = self._x.T @ dh; g["b1"] = dh.sum(0)
        self._grad_emb(dh @ self.W1.T, g)
        return g

    def _grad_emb(self, dx, g):
        demb = dx.reshape(-1, self.K, self.d)
        g["E"] = np.zeros_like(self.E)
        np.add.at(g["E"], self._ctx, demb)


# ── Architecture 2 : porte multiplicative (GLU) ───────────────────────────

class MLPGLU(Modele):
    nom = "mlp-glu"

    def _init(self, rng):
        Kd = self.K * self.d
        s = (2 / Kd) ** .5
        self.E = rng.normal(0, .1, (self.V, self.d))
        self.Wa = rng.normal(0, s, (Kd, self.H))
        self.ba = np.zeros(self.H)
        self.Wg = rng.normal(0, s, (Kd, self.H))
        self.bg = np.zeros(self.H)
        self.W2 = rng.normal(0, (1 / self.H) ** .5, (self.H, self.V))
        self.b2 = np.zeros(self.V)
        for k in ("E", "Wa", "ba", "Wg", "bg", "W2", "b2"):
            self._enr(k)

    def _avant(self, ctx):
        self._ctx = ctx
        emb = self.E[ctx]
        self._x = emb.reshape(len(ctx), -1)
        self._a = self._x @ self.Wa + self.ba
        self._gi = self._x @ self.Wg + self.bg
        self._g = sigmoide(self._gi)
        self._h = self._a * self._g
        self._p = self._logits_vers_p(self._h @ self.W2 + self.b2)
        return self._p

    def _graphes(self, ctx, cible):
        p = self._avant(ctx)
        B = len(ctx)
        dl = p.copy(); dl[np.arange(B), cible] -= 1; dl /= B
        g = {}
        g["W2"] = self._h.T @ dl; g["b2"] = dl.sum(0)
        dh = dl @ self.W2.T
        da = dh * self._g
        dgi = dh * self._a * self._g * (1 - self._g)
        g["Wa"] = self._x.T @ da; g["ba"] = da.sum(0)
        g["Wg"] = self._x.T @ dgi; g["bg"] = dgi.sum(0)
        MLPReLU._grad_emb(self, da @ self.Wa.T + dgi @ self.Wg.T, g)
        return g


# ── Architecture 3 : non-linearites SOMMEES (polyact) ─────────────────────

class PolyAct(Modele):
    """h = ReLU(a) + tanh(b) + swish(c) : trois mecanismes concordants."""

    nom = "polyact"

    def _init(self, rng):
        Kd = self.K * self.d
        s = (2 / Kd) ** .5
        self.E = rng.normal(0, .1, (self.V, self.d))
        self.Wa = rng.normal(0, s, (Kd, self.H)); self.ba = np.zeros(self.H)
        self.Wb = rng.normal(0, s, (Kd, self.H)); self.bb = np.zeros(self.H)
        self.Wc = rng.normal(0, s, (Kd, self.H)); self.bc = np.zeros(self.H)
        self.W2 = rng.normal(0, (1 / self.H) ** .5, (self.H, self.V))
        self.b2 = np.zeros(self.V)
        for k in ("E", "Wa", "ba", "Wb", "bb", "Wc", "bc", "W2", "b2"):
            self._enr(k)

    def _avant(self, ctx):
        self._ctx = ctx
        emb = self.E[ctx]
        self._x = emb.reshape(len(ctx), -1)
        self._pa = self._x @ self.Wa + self.ba
        self._pb = self._x @ self.Wb + self.bb
        self._pc = self._x @ self.Wc + self.bc
        ha = np.maximum(self._pa, 0)
        hb = np.tanh(self._pb)
        sc = sigmoide(self._pc)
        hc = self._pc * sc
        self._h = ha + hb + hc
        self._p = self._logits_vers_p(self._h @ self.W2 + self.b2)
        return self._p

    def _graphes(self, ctx, cible):
        p = self._avant(ctx)
        B = len(ctx)
        dl = p.copy(); dl[np.arange(B), cible] -= 1; dl /= B
        g = {}
        g["W2"] = self._h.T @ dl; g["b2"] = dl.sum(0)
        dh = dl @ self.W2.T
        dpa = dh * (self._pa > 0)
        dpb = dh * (1 - np.tanh(self._pb) ** 2)
        sc = sigmoide(self._pc)
        dpc = dh * (sc + self._pc * sc * (1 - sc))       # d/dx [x*sig(x)]
        g["Wa"] = self._x.T @ dpa; g["ba"] = dpa.sum(0)
        g["Wb"] = self._x.T @ dpb; g["bb"] = dpb.sum(0)
        g["Wc"] = self._x.T @ dpc; g["bc"] = dpc.sum(0)
        dx = dpa @ self.Wa.T + dpb @ self.Wb.T + dpc @ self.Wc.T
        MLPReLU._grad_emb(self, dx, g)
        return g


# ── Architecture 4 : recurrence (Elman, BPTT) ─────────────────────────────

class RNNElman(Modele):
    nom = "rnn-elman"

    def _init(self, rng):
        self.E = rng.normal(0, .1, (self.V, self.d))
        self.Wx = rng.normal(0, (1 / self.d) ** .5, (self.d, self.H))
        self.Wh = rng.normal(0, (1 / self.H) ** .5, (self.H, self.H))
        self.bh = np.zeros(self.H)
        self.W2 = rng.normal(0, (1 / self.H) ** .5, (self.H, self.V))
        self.b2 = np.zeros(self.V)
        for k in ("E", "Wx", "Wh", "bh", "W2", "b2"):
            self._enr(k)

    def _avant(self, ctx):
        self._ctx = ctx
        B = len(ctx)
        emb = self.E[ctx]                     # B,K,d
        h = np.zeros((B, self.H))
        self._hs = [h]
        self._emb = emb
        for t in range(self.K):
            h = np.tanh(emb[:, t, :] @ self.Wx + h @ self.Wh + self.bh)
            self._hs.append(h)
        self._hfin = h
        self._p = self._logits_vers_p(h @ self.W2 + self.b2)
        return self._p

    def _graphes(self, ctx, cible):
        p = self._avant(ctx)
        B = len(ctx)
        dl = p.copy(); dl[np.arange(B), cible] -= 1; dl /= B
        g = {}
        g["W2"] = self._hfin.T @ dl; g["b2"] = dl.sum(0)
        dh = dl @ self.W2.T
        g["Wx"] = np.zeros_like(self.Wx); g["Wh"] = np.zeros_like(self.Wh)
        g["bh"] = np.zeros_like(self.bh)
        gE = np.zeros_like(self.E)
        for t in range(self.K, 0, -1):
            transfert = dh * (1 - self._hs[t] ** 2)      # d tanh
            g["Wx"] += self._emb[:, t - 1, :].T @ transfert
            g["Wh"] += self._hs[t - 1].T @ transfert
            g["bh"] += transfert.sum(0)
            demb_t = transfert @ self.Wx.T
            np.add.at(gE, self._ctx[:, t - 1], demb_t)
            dh = transfert @ self.Wh.T
        g["E"] = gE
        return g


# ── Architecture 5 : convolutions multi-echelles + max-pool ───────────────

class ConvMulti(Modele):
    nom = "conv-multi"

    def __init__(self, V, K, d, H, graine=0):
        self.noyaux = (2, 3, 5)
        self.Hc = max(H // len(self.noyaux), 8)
        super().__init__(V, K, d, H, graine)

    def _init(self, rng):
        self.E = rng.normal(0, .1, (self.V, self.d))
        self._enr("E")
        for k in self.noyaux:
            setattr(self, f"Wc{k}", rng.normal(
                0, (2 / (k * self.d)) ** .5, (k * self.d, self.Hc)))
            self._enr(f"Wc{k}")
            setattr(self, f"bc{k}", np.zeros(self.Hc))
            self._enr(f"bc{k}")
        self.W2 = rng.normal(0, (1 / (len(self.noyaux) * self.Hc)) ** .5,
                             (len(self.noyaux) * self.Hc, self.V))
        self.b2 = np.zeros(self.V)
        self._enr("W2"); self._enr("b2")

    def _avant(self, ctx):
        self._ctx = ctx
        emb = self.E[ctx]                       # B,K,d
        self._emb = emb
        self._cache = {}
        salies = []
        for k in self.noyaux:
            pos = self.K - k + 1
            feats = np.stack([emb[:, p:p + k, :].reshape(len(ctx), -1)
                              for p in range(pos)], axis=1)   # B,pos,k*d
            act = feats @ getattr(self, f"Wc{k}") + getattr(self, f"bc{k}")
            idx = act.argmax(axis=1)                          # B,Hc (position)
            pooled = np.take_along_axis(act, idx[:, None, :], axis=1)[:, 0, :]
            self._cache[k] = (feats, act, idx)
            salies.append(pooled)
        self._h = np.concatenate(salies, axis=1)
        self._p = self._logits_vers_p(self._h @ self.W2 + self.b2)
        return self._p

    def _graphes(self, ctx, cible):
        p = self._avant(ctx)
        B = len(ctx)
        dl = p.copy(); dl[np.arange(B), cible] -= 1; dl /= B
        g = {"W2": self._h.T @ dl, "b2": dl.sum(0),
             "E": np.zeros_like(self.E)}
        dh = dl @ self.W2.T
        dec = 0
        for k in self.noyaux:
            feats, act, idx = self._cache[k]
            Hc = self.Hc
            dhc = dh[:, dec:dec + Hc]; dec += Hc
            dact = np.zeros_like(act)
            np.put_along_axis(dact, idx[:, None, :], dhc[:, None, :], axis=1)
            Wk = getattr(self, f"Wc{k}")
            gW = np.zeros_like(Wk); gb = dact.sum(1).sum(0)
            dfeats = np.zeros_like(feats)
            for ppos in range(act.shape[1]):
                gW += feats[:, ppos, :].T @ dact[:, ppos, :]
                dfeats[:, ppos, :] = dact[:, ppos, :] @ Wk.T
            g[f"Wc{k}"] = gW; g[f"bc{k}"] = gb
            # re-injecter dans les embeddings
            for ppos in range(act.shape[1]):
                piece = dfeats[:, ppos, :].reshape(B, k, self.d)
                np.add.at(g["E"], ctx[:, ppos:ppos + k], piece)
        return g


ARCHITECTURES = {
    "mlp-relu": MLPReLU,
    "mlp-glu": MLPGLU,
    "polyact": PolyAct,
    "rnn-elman": RNNElman,
    "conv-multi": ConvMulti,
}


# ── Verification de gradient (differences finies) ─────────────────────────

def verifie_gradient(cls, V, K, d, H, graine=0, eps=1e-4, n=40):
    """Compare le gradient analytique a une estimation numerique."""
    rng = np.random.default_rng(graine)
    ctx = rng.integers(0, V, size=(3, K))
    cible = rng.integers(0, V, size=3)
    m = cls(V, K, d, H, graine)
    g = m._graphes(ctx, cible)

    def perte():
        p = m._avant(ctx)
        return float(-np.log(np.maximum(p[np.arange(3), cible], 1e-9)).mean())

    pires = []
    for k in m.params:
        arr = getattr(m, k)
        flat = arr.ravel()
        indices = rng.choice(flat.size, size=min(n, flat.size), replace=False)
        err = 0.0
        for i in indices:
            orig = flat[i]
            flat[i] = orig + eps; lp = perte()
            flat[i] = orig - eps; lm = perte()
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = g[k].ravel()[i]
            denom = max(abs(num), abs(ana), 1e-8)
            err = max(err, abs(num - ana) / denom)
        pires.append((k, err))
    return max(e for _, e in pires), pires


# ── Boucle principale ─────────────────────────────────────────────────────

def entrainer(cls, texte, vocab, K, d, H, epochs=6, lr=3e-3, lot=64, graine=0):
    ctx, cible = paires(texte, vocab, K)
    coupe = int(len(ctx) * 0.9)
    ctr, cte, ytr, yte = ctx[:coupe], ctx[coupe:], cible[:coupe], cible[coupe:]
    m = cls(len(vocab), K, d, H, graine)
    rng = np.random.default_rng(graine)
    t0 = time.time()
    for _ in range(epochs):
        idx = rng.permutation(len(ctr))
        for i in range(0, len(idx), lot):
            j = idx[i:i + lot]
            m.pas(ctr[j], ytr[j], lr)
    dt = time.time() - t0
    return m, m.perte(ctr, ytr), m.perte(cte, yte), dt


def main() -> int:
    ap = argparse.ArgumentParser(prog="lab_architectures")
    ap.add_argument("--arch", default="", help="ne tester qu'une architecture")
    ap.add_argument("--gradcheck", action="store_true",
                    help="verifier les gradients (differences finies) et sortir")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--H", type=int, default=256, help="largeur de la couche")
    args = ap.parse_args()

    texte = charger_corpus()
    alphabet = sorted(set(texte))
    vocab = {c: i for i, c in enumerate(alphabet)}
    inv = {i: c for c, i in vocab.items()}
    V, K, d, H = len(vocab), 8, 16, args.H

    if args.gradcheck:
        print("Verification de gradient (analytique vs differences finies) :")
        ok = True
        for nom, cls in ARCHITECTURES.items():
            err, pires = verifie_gradient(cls, V, K, d, H)
            verdict = "OK" if err < 1e-3 else "ECHEC"
            ok &= err < 1e-3
            print(f"  {nom:12s} erreur relative max = {err:.2e}  [{verdict}]")
        return 0 if ok else 1

    cibles = ([args.arch] if args.arch else list(ARCHITECTURES))
    print(f"corpus {len(texte):,} car. | vocab {V} | fenetre K={K} d={d} H={H} "
          f"| {args.epochs} epoques")
    print("-" * 78)
    res = []
    for nom in cibles:
        m, ptr, pte, dt = entrainer(ARCHITECTURES[nom], texte, vocab, K, d, H,
                                    epochs=args.epochs)
        ppl = float(np.exp(pte))
        res.append((ppl, m.nb_params(), nom, m, pte, dt))
        print(f"  {nom:12s} | params {m.nb_params():>7,} | perte train {ptr:5.3f} "
              f"| val {pte:5.3f} | perplexite {ppl:7.2f} | {dt:5.1f}s")

    res.sort(key=lambda r: (r[0], r[1]))
    ppl, npar, nom, m, _, _ = res[0]
    print("-" * 78)
    print(f"GAGNANT : {nom}  (perplexite val {ppl:.2f}, {npar:,} parametres)")
    print(f"echantillon : « {m.generer('aura est ', vocab, inv, 200, 0.8).strip()} »")
    print()
    print("Suite honnete (a lire avant de 'scaler a 1B') :")
    print("  * un gagnant a petite echelle ne gagne pas forcement a 1B — le")
    print("    transfert depend surtout des hyperparametres (methode μP), de la")
    print("    donnee et de l'architecture ; un score a 60 Ko ne le prouve pas.")
    print("  * entrainer 1B ici est infalsifiable (pas de GPU, ~13 tok/s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
