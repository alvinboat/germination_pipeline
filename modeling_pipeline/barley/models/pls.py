"""PLS discriminant analysis.

PLS-DA is PLS regression against a one-hot target, with the predicted class
being the argmax over columns. It is the standard workhorse for NIR spectra and,
at this sample size, the model to beat: 548 kernels over 25 dishes is not much
data for anything with millions of parameters.

The number of components is the one hyperparameter that matters, and it is
chosen by an inner CV *inside each training fold*. Choosing it on the test fold
is the classic way to publish an inflated number.
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression

DEFAULT_GRID = tuple(range(2, 26, 2))


def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes), np.float64)
    Y[np.arange(len(y)), y] = 1.0
    return Y


class PLSDA:
    """PLS regression on one-hot targets; predict = argmax."""

    def __init__(self, n_components=10, n_classes=None):
        self.n_components = n_components
        self.n_classes = n_classes
        self.model = None

    def fit(self, X, y):
        self.n_classes = self.n_classes or int(y.max()) + 1
        n = min(self.n_components, X.shape[1], max(len(X) - 1, 1))
        self.model = PLSRegression(n_components=n, scale=False)
        self.model.fit(X, one_hot(y, self.n_classes))
        return self

    def decision(self, X):
        return self.model.predict(X)

    def predict(self, X):
        return np.asarray(self.decision(X)).argmax(1)


def balanced_accuracy(y_true, y_pred, n_classes):
    accs = []
    for c in range(n_classes):
        m = y_true == c
        if m.any():
            accs.append(float((y_pred[m] == c).mean()))
    return float(np.mean(accs)) if accs else 0.0


def choose_components(X, y, groups, n_classes, grid=DEFAULT_GRID, n_inner=4, seed=0):
    """Inner grouped CV over `grid`. -> (best n_components, {n: score}).

    The inner split groups by the same column as the outer one, so component
    selection is never helped by leakage the outer split forbids.
    """
    uniq = np.array(sorted(set(groups.tolist())), dtype=object)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    assign = {uniq[i]: k % n_inner for k, i in enumerate(order)}
    fold_of = np.array([assign[g] for g in groups.tolist()])

    scores = {}
    for n in grid:
        if n > X.shape[1]:
            continue
        got = []
        for k in range(n_inner):
            tr, te = fold_of != k, fold_of == k
            if not te.any() or len(set(y[tr].tolist())) < 2:
                continue
            m = PLSDA(n, n_classes).fit(X[tr], y[tr])
            got.append(balanced_accuracy(y[te], m.predict(X[te]), n_classes))
        if got:
            scores[n] = float(np.mean(got))
    if not scores:
        return min(grid), {}
    best = max(scores, key=lambda n: (scores[n], -n))   # ties -> fewer components
    return best, scores
