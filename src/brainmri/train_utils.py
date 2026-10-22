"""Обучение без torch: выбор примеров по эпохам, метрики, доверительные интервалы, расписание lr."""
from __future__ import annotations
import math
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


class EpochSampler:
    """На каждой эпохе: классы берутся поровну, внутри класса выбирается кластер (уникальный снимок),
    внутри кластера — случайный файл. Так размноженные снимки не перетягивают обучение.

    hard_weight > 0 повышает шанс кластеров, где простой «шорткат» ошибается (sc_p_true мала).
    """

    def __init__(self, members: dict, cl_cls: dict, cl_p: dict | None = None, hard_weight: float = 0.0, seed: int = 42):
        self.members = members
        self.seed = seed
        self.by_cls = {c: [k for k, v in cl_cls.items() if v == c] for c in CLASSES}
        self.w = {}
        for c, ids in self.by_cls.items():
            if not ids:
                raise ValueError(f"Нет кластеров класса {c}")
            p = np.array([self._p(cl_p, k) for k in ids], dtype=float)
            w = 1.0 + hard_weight * (1.0 - p)
            self.w[c] = w / w.sum()

    @staticmethod
    def _p(cl_p, k):
        v = None if cl_p is None else cl_p.get(k)
        return 1.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

    def draw(self, epoch: int, n: int):
        rng = np.random.default_rng((self.seed, epoch))
        per = n // len(CLASSES)
        items = []
        for k, c in enumerate(CLASSES):
            ids = self.by_cls[c]
            for j in rng.choice(len(ids), size=per, replace=True, p=self.w[c]):
                m = self.members[ids[j]]
                items.append((int(m[rng.integers(len(m))]), k))
        order = rng.permutation(len(items))
        return [items[i] for i in order]


def compute_metrics(y, proba) -> dict:
    """y: индексы классов, proba: (n, 4). Только обычные python-типы (чтобы сохранять в json)."""
    y = np.asarray(y)
    pred = np.asarray(proba).argmax(1)
    out = dict(n=int(len(y)),
               acc=float(accuracy_score(y, pred)),
               macro_f1=float(f1_score(y, pred, average="macro", labels=list(range(len(CLASSES))), zero_division=0)),
               recall=[float(v) for v in recall_score(y, pred, labels=list(range(len(CLASSES))), average=None, zero_division=0)])
    nt = CLASSES.index("notumor")
    try:
        out["auc_tumor"] = float(roc_auc_score(y != nt, 1 - np.asarray(proba)[:, nt]))
    except ValueError:
        out["auc_tumor"] = None
    return out


def bootstrap_ci(y, pred, fn, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Доверительный интервал метрики fn(y_true, y_pred) бутстрэпом по снимкам."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(fn(y[i], pred[i]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def cosine_warmup(step: int, total: int, warm: int) -> float:
    """Множитель lr: линейный разогрев warm шагов, затем косинус до ~0."""
    if step < warm:
        return (step + 1) / max(1, warm)
    prog = (step - warm) / max(1, total - warm)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
