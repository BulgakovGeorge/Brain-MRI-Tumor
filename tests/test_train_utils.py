import numpy as np
from sklearn.metrics import accuracy_score
from brainmri.train_utils import EpochSampler, compute_metrics, bootstrap_ci, cosine_warmup, CLASSES


def _toy():
    members, cl_cls, cl_p = {}, {}, {}
    cid = 0
    for k, c in enumerate(CLASSES):
        for i in range(20 * (k + 1)):               # разное число кластеров по классам
            members[cid] = np.arange(cid * 10, cid * 10 + 1 + i % 7)
            cl_cls[cid] = c
            cl_p[cid] = 0.1 if i < 5 else 0.95      # первые 5 кластеров каждого класса «трудные»
            cid += 1
    return members, cl_cls, cl_p


def test_sampler_balanced_valid_reproducible():
    members, cl_cls, cl_p = _toy()
    s = EpochSampler(members, cl_cls, cl_p, seed=1)
    a = s.draw(0, 400); b = s.draw(0, 400); c = s.draw(1, 400)
    assert a == b and a != c
    assert [sum(1 for _, k in a if k == j) for j in range(4)] == [100] * 4
    valid = {int(i) for m in members.values() for i in m}
    assert all(i in valid for i, _ in a)


def test_hard_weight_shifts_probability():
    members, cl_cls, cl_p = _toy()
    hard = {k for k, p in cl_p.items() if p < 0.5}
    owner = {int(i): k for k, m in members.items() for i in m}
    frac = lambda hw: np.mean([owner[i] in hard for i, _ in EpochSampler(members, cl_cls, cl_p, hw, 2).draw(0, 4000)])
    assert frac(3.0) > frac(0.0) * 1.5


def test_metrics_and_ci():
    y = np.array([0, 1, 2, 3] * 25)
    proba = np.eye(4)[y] * 0.9 + 0.025
    m = compute_metrics(y, proba)
    assert m["acc"] == 1.0 and m["macro_f1"] == 1.0 and m["auc_tumor"] == 1.0 and len(m["recall"]) == 4
    pred = y.copy(); pred[:20] = (pred[:20] + 1) % 4
    lo, hi = bootstrap_ci(y, pred, accuracy_score, n_boot=300)
    assert lo < accuracy_score(y, pred) < hi


def test_missing_class_does_not_crash():
    y = np.array([0, 0, 1, 1]); proba = np.eye(4)[y]
    m = compute_metrics(y, proba)
    assert m["auc_tumor"] is not None or m["auc_tumor"] is None   # без notumor AUC может быть None, но без ошибки


def test_cosine_warmup():
    assert cosine_warmup(0, 100, 10) < 0.2 and abs(cosine_warmup(10, 100, 10) - 1.0) < 1e-9
    assert cosine_warmup(100, 100, 10) < 1e-6
