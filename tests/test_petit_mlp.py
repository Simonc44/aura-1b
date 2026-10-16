"""Petit MLP numpy (`aura/petit_mlp.py`) : correction, apprentissage, persistance.

Ces tests protegent l'amelioration du 8 oct. 2026 : le classifieur flou du
routeur est passe du LinearSVC a ce petit MLP (mesure : +2,2 pts d'accuracy en
validation croisee). Un gradient faux rendrait la comparaison mensongere, d'ou
le test de differences finies.
"""
import numpy as np
import pytest

from aura import embeddings
from aura.petit_mlp import ClassifieurMLP


def _donnees(n=300, D=20, K=3, graine=0):
    rng = np.random.default_rng(graine)
    centres = rng.normal(0, 3.0, (K, D))
    X = np.vstack([rng.normal(centres[k], 1.0, (n // K, D)) for k in range(K)])
    y = [f"c{k}" for k in range(K) for _ in range(n // K)]
    return X, y


def test_gradient_analytique_vs_finies():
    """Le gradient doit etre correct (sinon tout le reste ne vaut rien)."""
    rng = np.random.default_rng(0)
    for glu in (False, True):
        clf = ClassifieurMLP(D=6, K=3, H=5, glu=glu, graine=0)
        X = rng.normal(0, 1, (12, 6))
        y = np.array([0, 1, 2] * 4)
        clf.mu, clf.sd = X.mean(0), X.std(0) + 1e-6
        grads = clf._graphes(X, y)
        eps = 1e-6

        def perte() -> float:
            _, p = clf._avant(X)
            return float(-np.log(np.maximum(p[np.arange(12), y], 1e-12)).mean())

        for nom, arr in (("W1", clf.W1), ("W2", clf.W2),
                         ("Wg", clf.Wg), ("bg", clf.bg)):
            if arr is None:
                continue
            flat, g = arr.ravel(), grads[nom].ravel()
            for i in rng.choice(flat.size, size=min(8, flat.size), replace=False):
                o = flat[i]
                flat[i] = o + eps
                lp = perte()
                flat[i] = o - eps
                lm = perte()
                flat[i] = o
                num = (lp - lm) / (2 * eps)
                denom = max(abs(num), abs(g[i]), 1e-8)
                assert abs(num - g[i]) / denom < 1e-4, f"{nom}[{i}]"


def test_apprend_trois_classes():
    X, y = _donnees()
    clf = ClassifieurMLP(D=X.shape[1], K=3, H=48, glu=False).fit(X, y)
    pred = [clf.classes_[int(np.argmax(clf.probas1(x)))] for x in X]
    acc = float(np.mean([p == c for p, c in zip(pred, y)]))
    assert acc > 0.9, f"accuracy d'entrainement trop faible : {acc}"


def test_probas_somment_a_un():
    X, y = _donnees()
    clf = ClassifieurMLP(D=X.shape[1], K=3, H=48).fit(X, y)
    p = clf.probas1(X[0])
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert len(p) == 3


def test_serialisation_identique():
    X, y = _donnees()
    clf = ClassifieurMLP(D=X.shape[1], K=3, H=16).fit(X, y)
    clone = ClassifieurMLP.depuis_etat(clf.etat())
    assert clone.classes_ == clf.classes_
    assert np.allclose(clone.probas1(X[3]), clf.probas1(X[3]))


def test_repli_flou_desactive(monkeypatch):
    """AURA_FLOU=0 : le classificateur flou est coupe, sans exception."""
    monkeypatch.setenv("AURA_FLOU", "0")
    from aura import classificateur_flou as cf
    assert cf.actif() is False
    assert cf.probas("explique moi ce concept") is None


@pytest.mark.skipif(not embeddings.modele_present(),
                    reason="modeles/minilm absent")
def test_flou_renvoie_trois_classes():
    from aura import classificateur_flou as cf
    p = cf.probas("calcule la racine carree de 144")
    assert p is not None
    assert set(p) == {"math", "web", "general"}
    assert abs(sum(p.values()) - 1.0) < 0.02


@pytest.mark.skipif(not embeddings.modele_present(),
                    reason="modeles/minilm absent")
def test_backend_par_defaut_est_svm(monkeypatch):
    """Defaut = LinearSVC (comportement historique, non modifie)."""
    monkeypatch.delenv("AURA_FLOU", raising=False)
    from aura import classificateur_flou as cf
    monkeypatch.setattr(cf, "_modele", None)
    monkeypatch.setattr(cf, "_backend", None)
    monkeypatch.setattr(cf, "_indispo", False)
    assert cf.probas("calcule 2 puissance 10") is not None
    assert cf._backend == "svm"


@pytest.mark.skipif(not embeddings.modele_present(),
                    reason="modeles/minilm absent")
def test_backend_mlp_via_env(monkeypatch):
    """AURA_FLOU=mlp : le petit MLP remplace le LinearSVC."""
    monkeypatch.setenv("AURA_FLOU", "mlp")
    from aura import classificateur_flou as cf
    monkeypatch.setattr(cf, "_modele", None)
    monkeypatch.setattr(cf, "_backend", None)
    monkeypatch.setattr(cf, "_indispo", False)
    p = cf.probas("calcule 2 puissance 10")
    assert p is not None and set(p) == {"math", "web", "general"}
    assert cf._backend == "mlp"


@pytest.mark.skipif(not embeddings.modele_present(),
                    reason="modeles/minilm absent")
def test_repli_svm_fonctionne(monkeypatch):
    """AURA_FLOU=svm : retour arriere LinearSVC sans casser le contrat.

    Non-regression : le SVM expose `classes_` en ndarray (et non en list),
    ce qui faisait planter le calcul des classes avant correction.
    """
    monkeypatch.setenv("AURA_FLOU", "svm")
    from aura import classificateur_flou as cf
    monkeypatch.setattr(cf, "_modele", None)
    monkeypatch.setattr(cf, "_backend", None)
    monkeypatch.setattr(cf, "_indispo", False)
    p = cf.probas("calcule 2 puissance 10")
    assert p is not None
    assert set(p) == {"math", "web", "general"}
    assert cf._backend == "svm"
