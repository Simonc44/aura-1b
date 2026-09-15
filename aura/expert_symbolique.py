"""Pilier logique : Expert Symbolique par Programmation Genetique (PGS, ~50M eq.).

Principe : au lieu de deviner un calcul par statistique (hallucination), on fait
EVOLUER des arbres d'equations (add, sub, mul, div, pow) jusqu'a ce que l'erreur
tombe sous le seuil. La formule trouvee est verifiable, exacte, imprimable.

Contrat de normalisation : X et y sont ramenes min-max dans [0, 1] avant la
recherche. gplearn borne ses constantes a [-1, 1] : sans cette normalisation,
une loi comme Y = X^3 / 215 est tout simplement irrepresentable pour lui.
La formule renvoyee s'applique donc aux variables normalisees :
    X' = (x - min(x)) / (max(x) - min(x))     Y' = (y - min(y)) / (max(y) - min(y))
Les predictions sont denormalisees pour calculer l'erreur en unites reelles.

Le `pow` est protege : exposant borne [-5, 5], |x|, inf/nan -> 1.0, pour ne
jamais faire exploser le CPU.
"""
import numpy as np
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor


def _pow_protege(x1, x2):
    """Puissance robuste : borne l'exposant, neutralise inf/nan."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        resultat = np.power(np.abs(x1), np.clip(x2, -5, 5))
    return np.where(np.isnan(resultat) | np.isinf(resultat), 1.0, resultat)


_POW = make_function(function=_pow_protege, name="pow", arity=2)

# operateurs autorises par l'evolution
JEU_OPERATEURS = ("add", "sub", "mul", "div", _POW)


class ExpertSymbolique:
    """Regression symbolique : decouvre la loi Y = f(X) exacte a partir d'exemples."""

    def __init__(self, population=1500, generations=20, precision=0.001, graine=42):
        self._cfg = dict(population_size=population, generations=generations,
                         stopping_criteria=precision, random_state=graine)
        self.programme = None
        self.erreur = None
        # bornes min-max memorisees pour denormaliser
        self._x_min = self._x_max = self._y_min = self._y_max = None

    def _normaliser(self, X, y):
        self._x_min, self._x_max = X.min(axis=0), X.max(axis=0)
        self._y_min, self._y_max = float(y.min()), float(y.max())
        etendue_x = np.where(self._x_max - self._x_min == 0, 1.0, self._x_max - self._x_min)
        etendue_y = (self._y_max - self._y_min) or 1.0
        return (X - self._x_min) / etendue_x, (y - self._y_min) / etendue_y

    def resoudre(self, X, y) -> str:
        """Fit X (n, d) -> y (n,) ; renvoie la formule lisible ex. 'mul(mul(X0, X0), X0)'.

        La formule s'applique aux variables min-max normalisees dans [0, 1].
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(X) < 3:
            return ""  # pas assez d'exemples pour evoluer serieusement
        Xn, Yn = self._normaliser(X, y)
        reg = SymbolicRegressor(function_set=JEU_OPERATEURS,
                                p_crossover=0.7, p_subtree_mutation=0.1,
                                p_hoist_mutation=0.05, p_point_mutation=0.1,
                                const_range=(-1.0, 1.0), n_jobs=1, **self._cfg)
        reg.fit(Xn, Yn)
        self.programme = reg._program
        # erreur en unites REELLES (predictions denormalisees)
        etendue_y = (self._y_max - self._y_min) or 1.0
        pred = np.asarray(reg.predict(Xn)) * etendue_y + self._y_min
        self.erreur = float(np.mean((pred - y) ** 2))
        return str(self.programme)

    @property
    def fiable(self) -> bool:
        """Vrai si l'erreur est proche de zero : la formule est une garantie, pas une supposition."""
        return self.erreur is not None and self.erreur < 1e-6
