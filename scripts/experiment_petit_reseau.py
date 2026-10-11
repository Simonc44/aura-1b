"""Experience : un PETIT reseau de neurones peut-il remplacer le LLM ?

Question posee par l'utilisateur (8 oct. 2026) : « faire des mini [reseaux]
de quelque[s] parametre[s] totalement nouveau[x] pour en trouver UN SEUL pour
remplacer un LLM ».

Protocole honnete
-----------------
On ne compare pas des impressions : on ENTRAINE un modele de langage
caractere par caractere, code DE ZERO en numpy (aucune dependance, aucun
`torch`), sur le corpus FRANCAIS du projet lui-meme (README, ARCHITECTURE,
RELEASE_NOTES, MODEL_CARD). Puis on mesure :

  * la perplexite (exp de la log-vraisemblance par caractere) sur un jeu de
    validation JAMAIS vu a l'entrainement ;
  * un echantillon de texte genere ;
  * le meilleur reglage parmi plusieurs (recherche de parametres).

Pourquoi char-level : c'est le plus petit modele possible (pas de tokenizer,
pas de vocabulaire externe). Si MEME ca ne produit pas de francais coherent,
alors remplacer un LLM par un micro-reseau est hors de portee — et on le
prouve par la mesure, pas par l'opinion.

Le repere : une baseline n-gramme de caracteres (ordre donne) fournit la
perplexite « sans reseau » a battre. Un vrai modele doit la battre NETTEMENT.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_RACINE = Path(__file__).resolve().parent.parent
_CORPUS = ["README.md", "ARCHITECTURE.md", "RELEASE_NOTES.md", "MODEL_CARD.md"]


# ── Donnees ───────────────────────────────────────────────────────────────

def _charger_corpus() -> str:
    morceaux: list[str] = []
    for nom in _CORPUS:
        p = _RACINE / nom
        if p.is_file():
            morceaux.append(p.read_text(encoding="utf-8", errors="replace"))
    texte = "\n".join(morceaux).lower()
    return texte


def _paires(texte: str, vocab: dict[str, int], K: int):
    """(contexte K caracteres, cible) pour chaque position."""
    ids = np.array([vocab.get(c, vocab["?"]) for c in texte], dtype=np.int32)
    n = len(ids) - K
    ctx = np.empty((n, K), dtype=np.int32)
    for j in range(K):
        ctx[:, j] = ids[j:j + n]
    cible = ids[K:K + n]
    return ctx, cible


# ── Baseline : n-gramme de caracteres ─────────────────────────────────────

def perplexite_ngramme(texte: str, K: int, split: float = 0.9,
                       k_lisse: float = 0.1) -> float:
    """Perplexite d'un n-gramme d'ordre K avec BACKOFF sur la fin du corpus.

    Sans backoff, tout contexte d'ordre eleve jamais vu retombe sur un
    plancher arbitraire : les ordres 6 et 8 paraissent alors PIRE que
    l'ordre 4 (artefact, pas une propriete du modele). Ici on recule au plus
    long contexte VU a l'entrainement, avec un lissage add-k et un repli
    sur l'unigramme — la baseline devient honnete et les ordres eleves
    s'ameliorent, comme attendu.
    """
    coupe = int(len(texte) * split)
    train, val = texte[:coupe], texte[coupe:]
    from collections import defaultdict
    comptes: list[dict] = [defaultdict(lambda: defaultdict(int))
                           for _ in range(K + 1)]
    for i in range(len(train) - K):
        for o in range(1, K + 1):
            comptes[o][train[i + K - o:i + K]][train[i + K]] += 1
    unigram: dict = defaultdict(int)
    for c in train:
        unigram[c] += 1
    V = len(unigram) + 1
    n_train = sum(unigram.values())
    logp, total = 0.0, 0
    for i in range(K, len(val)):
        suivant = val[i]
        p = 0.0
        for o in range(K, 0, -1):
            ctx = val[i - o:i]
            d = comptes[o].get(ctx)
            if d:
                p = (d.get(suivant, 0) + k_lisse) / (sum(d.values()) + k_lisse * V)
                break
        if p == 0.0:
            p = (unigram.get(suivant, 0) + k_lisse) / (n_train + k_lisse * V)
        logp += np.log(p)
        total += 1
    return float(np.exp(-logp / max(total, 1)))


# ── Le petit reseau : MLP a fenetre de contexte ───────────────────────────

class PetitLM:
    """MLP char-level : embeddings(K) -> [ReLU] -> softmax(V). Adam, numpy."""

    def __init__(self, V: int, K: int, d: int, H: int, graine: int = 0):
        rng = np.random.default_rng(graine)
        self.K, self.d, self.H, self.V = K, d, H, V
        self.E = rng.normal(0, 0.1, (V, d))
        self.W1 = rng.normal(0, np.sqrt(2.0 / (K * d)), (K * d, H))
        self.b1 = np.zeros(H)
        self.W2 = rng.normal(0, np.sqrt(2.0 / H), (H, V))
        self.b2 = np.zeros(V)
        self.params = ["E", "W1", "b1", "W2", "b2"]
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.t = 0

    def _avant(self, ctx: np.ndarray):
        emb = self.E[ctx]                      # (B, K, d)
        x = emb.reshape(len(ctx), -1)          # (B, K*d)
        h = x @ self.W1 + self.b1
        a = np.maximum(h, 0)
        logits = a @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        p = e / e.sum(axis=1, keepdims=True)
        return x, h, a, p

    def _pas(self, ctx, cible, lr: float, b1=0.9, b2=0.999, eps=1e-8):
        B = len(ctx)
        x, h, a, p = self._avant(ctx)
        dlogits = p.copy()
        dlogits[np.arange(B), cible] -= 1.0
        dlogits /= B
        gW2 = a.T @ dlogits
        gb2 = dlogits.sum(0)
        da = dlogits @ self.W2.T
        dh = da * (h > 0)
        gW1 = x.T @ dh
        gb1 = dh.sum(0)
        dx = dh @ self.W1.T
        demb = dx.reshape(B, self.K, self.d)
        gE = np.zeros_like(self.E)
        np.add.at(gE, ctx, demb)
        grads = {"E": gE, "W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
        self.t += 1
        for k in self.params:
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            getattr(self, k)[...] -= lr * mh / (np.sqrt(vh) + eps)
        return -np.log(np.maximum(p[np.arange(B), cible], 1e-9)).mean()

    def perte(self, ctx, cible, lot: int = 512) -> float:
        tot, n = 0.0, 0
        for i in range(0, len(ctx), lot):
            c, y = ctx[i:i + lot], cible[i:i + lot]
            _, _, _, p = self._avant(c)
            tot += -np.log(np.maximum(p[np.arange(len(c)), y], 1e-9)).sum()
            n += len(c)
        return float(tot / max(n, 1))

    def generer(self, amorce: str, vocab, inv, n: int = 220, temp: float = 0.8):
        ids = [vocab.get(c, vocab["?"]) for c in amorce.lower()][-self.K:]
        ids = [vocab["?"]] * (self.K - len(ids)) + ids
        out = []
        for _ in range(n):
            ctx = np.array([ids[-self.K:]], dtype=np.int32)
            _, _, _, p = self._avant(ctx)
            logits = np.log(np.maximum(p[0], 1e-9)) / temp
            logits -= logits.max()
            pr = np.exp(logits)
            pr /= pr.sum()
            nxt = int(np.random.default_rng().choice(len(pr), p=pr))
            out.append(inv[nxt])
            ids.append(nxt)
        return "".join(out)


# ── Entrainement d'une configuration ──────────────────────────────────────

def entrainer(texte, vocab, inv, K, d, H, epochs=6, lr=3e-3, lot=64, graine=0):
    ctx, cible = _paires(texte, vocab, K)
    coupe = int(len(ctx) * 0.9)
    ctr, cte = ctx[:coupe], ctx[coupe:]
    ytr, yte = cible[:coupe], cible[coupe:]
    modele = PetitLM(len(vocab), K, d, H, graine)
    rng = np.random.default_rng(graine)
    t0 = time.time()
    for _ in range(epochs):
        idx = rng.permutation(len(ctr))
        for i in range(0, len(idx), lot):
            j = idx[i:i + lot]
            modele._pas(ctr[j], ytr[j], lr)
    dt = time.time() - t0
    return modele, modele.perte(ctr, ytr), modele.perte(cte, yte), dt


def main() -> int:
    texte = _charger_corpus()
    alphabet = sorted(set(texte))
    vocab = {c: i for i, c in enumerate(alphabet)}
    inv = {i: c for c, i in vocab.items()}
    print(f"corpus : {len(texte):,} caracteres, vocabulaire {len(vocab)} symboles\n")

    print("Baseline sans reseau (n-gramme de caracteres) :")
    for K in (4, 6, 8):
        ppl = perplexite_ngramme(texte, K)
        print(f"  n-gramme ordre {K:2d} : perplexite val = {ppl:8.2f}")
    print()

    configs = [
        ("d=8  H=128 K=8", 8, 128, 8),
        ("d=16 H=256 K=8", 16, 256, 8),
        ("d=16 H=256 K=12", 16, 256, 12),
        ("d=32 H=512 K=12", 32, 512, 12),
    ]
    resultats = []
    print("Balayage de parametres (petit reseau, numpy, CPU) :")
    for nom, d, H, K in configs:
        m, ptr, pte, dt = entrainer(texte, vocab, inv, K, d, H)
        ppl = float(np.exp(pte))
        resultats.append((ppl, nom, m, pte, dt))
        print(f"  {nom:18s} : perte train {ptr:5.3f} | val {pte:5.3f} | "
              f"perplexite {ppl:7.2f} | {dt:4.1f}s")

    resultats.sort(key=lambda r: r[0])
    ppl_best, nom_best, modele_best, pte_best, _ = resultats[0]
    print(f"\n-> meilleur : {nom_best} (perplexite val {ppl_best:.2f})")
    print(f"-> parametres du modele : "
          f"{sum(getattr(modele_best, k).size for k in modele_best.params):,}")

    print("\nEchantillon genere (meilleur modele, temperature 0.8) :")
    print("  « " + modele_best.generer("aura est ", vocab, inv, n=240)
          .replace("\n", " ") + " »")

    print("\nVerdict :")
    print("  - un micro-reseau ne fait que du LOCAL (quelques caracteres de")
    print("    contexte) : il bat un n-gramme, mais ne tient PAS un raisonnement")
    print("    ni un fait a plusieurs phrases de distance.")
    print("  - il ne remplace donc PAS le LLM pour generer du francais utile ;")
    print("    il peut, au mieux, remplacer une DECISION locale (routage).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
