"""Scoring, and the two controls that keep a score honest.

Plain accuracy is the wrong headline here. Classes are near-balanced for variety
but will not be for germination, and the interesting failure is a model that
looks good because it learned the plate. So the reported number is balanced
accuracy, and it comes with:

  permutation floor   the same pipeline on shuffled labels. NOT 1/n_classes:
                      with grouped folds and correlated views the real chance
                      level sits above it, and the gap between a score and this
                      floor is the only meaningful claim.
  dish-identity       how well the same features predict which DISH a view came
                      from. Variety is perfectly confounded with dish, so if
                      dish is as predictable as variety, the model is reading
                      the plate rather than the grain.
"""
import numpy as np


def balanced_accuracy(y_true, y_pred, n_classes):
    accs = []
    for c in range(n_classes):
        m = y_true == c
        if m.any():
            accs.append(float((y_pred[m] == c).mean()))
    return float(np.mean(accs)) if accs else 0.0


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def summarise(y_true, y_pred, n_classes):
    return {
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, n_classes),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, n_classes),
        "n": int(len(y_true)),
    }


def aggregate_by_kernel(kernels, y_true, scores):
    """Average per-view decision scores within a kernel -> one prediction each.

    A kernel is measured 8 times (2 modes x 2 sides x 2 hours). Those views are
    not independent evidence about different grains, they are repeat looks at
    the same one, so a per-kernel number answers "can we call this kernel?"
    while the per-view number answers "can we call this measurement?".
    """
    out_k, out_t, out_s = [], [], []
    for k in sorted(set(kernels.tolist())):
        m = kernels == k
        t = y_true[m]
        if len(set(t.tolist())) != 1:
            raise SystemExit(f"kernel {k} carries more than one label")
        out_k.append(k)
        out_t.append(int(t[0]))
        out_s.append(np.asarray(scores)[m].mean(0))
    return (np.array(out_k, object), np.array(out_t),
            np.asarray(out_s).argmax(1), np.asarray(out_s))


def format_confusion(cm, classes):
    w = max(len(c) for c in classes) + 1
    head = " " * (w + 2) + " ".join(f"{c[:6]:>6s}" for c in classes)
    lines = [head]
    for i, c in enumerate(classes):
        lines.append(f"{c:>{w}s}  " + " ".join(f"{v:6d}" for v in cm[i]))
    return "\n".join(lines)


# ------------------------------------------------------ binary, imbalanced --
# The germination call is 89/11, which breaks every habit that works on the
# near-balanced variety task. Accuracy is 0.89 for a model that says "yes" to
# everything; argmax over a one-hot PLS fit does almost exactly that. So the
# headline is threshold-free (AUC), the minority class gets its own average
# precision quoted against its prevalence, and the hard call uses a threshold
# tuned inside the training fold rather than an implicit 0.5.

def auc(y_true, score):
    """Rank AUC (Mann-Whitney). None if only one class is present.

    Ties count a half, which matters: a PLS score can tie exactly across a fold
    and the optimistic convention would quietly credit the model for it.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score, float)
    pos, neg = score[y_true == 1], score[y_true == 0]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    srt = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(srt):                      # average ranks within a tie run
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def grouped_auc(y_true, score, groups):
    """AUC over pairs drawn from the SAME group only. None if no such pair.

    The strict control. Pooled AUC can be earned entirely by ranking dishes, or
    varieties, against each other -- both of which a model can do off the plate.
    Restricting the comparison to kernels that shared a dish asks whether the
    spectra rank grain within one plate, which no plate signature can fake.
    """
    y_true, score = np.asarray(y_true), np.asarray(score, float)
    groups = np.asarray(groups)
    num = den = 0.0
    for g in np.unique(groups):
        m = groups == g
        a = auc(y_true[m], score[m])
        if a is None:
            continue
        n = int((y_true[m] == 1).sum()) * int((y_true[m] == 0).sum())
        num += a * n
        den += n
    return float(num / den) if den else None


def average_precision(y_true, score, positive=1):
    """AP for one class. Its chance level IS that class's prevalence."""
    y = (np.asarray(y_true) == positive).astype(int)
    s = np.asarray(score, float) * (1 if positive == 1 else -1)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def roc_curve(y_true, score):
    """-> (fpr, tpr) including the (0,0) and (1,1) endpoints."""
    y, s = np.asarray(y_true), np.asarray(score, float)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    P, N = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    return (np.concatenate([[0], fp / N, [1]]),
            np.concatenate([[0], tp / P, [1]]))


def best_threshold(y_true, score):
    """Threshold maximising balanced accuracy. -> (threshold, that accuracy).

    Swept over midpoints between observed scores, so the choice does not depend
    on the scale PLS happens to put its output on.
    """
    y, s = np.asarray(y_true), np.asarray(score, float)
    if len(set(y.tolist())) < 2:
        return float(np.median(s)), 0.5
    cand = np.unique(s)
    mids = np.concatenate([[cand[0] - 1], (cand[:-1] + cand[1:]) / 2, [cand[-1] + 1]])
    best, best_t = -1.0, float(mids[0])
    for t in mids:
        p = (s >= t).astype(int)
        ba = balanced_accuracy(y, p, 2)
        if ba > best:
            best, best_t = ba, float(t)
    return best_t, float(best)


def binary_summary(y_true, score, threshold, minority=0, pred=None):
    """Threshold-free metrics from `score`; hard-call metrics from `pred`.

    `pred` matters more than it looks. Each CV fold fits its own model AND its
    own threshold, so the fold's hard calls are the only ones that fold's model
    would actually have made. Pooling the scores and re-thresholding them at
    some single value afterwards borrows information across folds and flatters
    the result -- on the reflectance-8h run it was worth four points of balanced
    accuracy. Pass the per-fold predictions; `threshold` is then recorded for
    reference only.
    """
    y, s = np.asarray(y_true), np.asarray(score, float)
    p = (s >= threshold).astype(int) if pred is None else np.asarray(pred)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return {
        "n": int(len(y)), "prevalence": float((y == 1).mean()),
        "auc": auc(y, s),
        "ap_germinated": average_precision(y, s, 1),
        "ap_never": average_precision(y, s, 0),
        "minority_prevalence": float((y == minority).mean()),
        "threshold": float(threshold),
        "balanced_accuracy": balanced_accuracy(y, p, 2),
        "accuracy": accuracy(y, p),
        "sensitivity": tp / (tp + fn) if tp + fn else None,   # germinated found
        "specificity": tn / (tn + fp) if tn + fp else None,   # duds found
        "precision_never": tn / (tn + fn) if tn + fn else None,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
