"""Кластеры «один и тот же снимок» и честное разбиение по кластерам.

Кластер объединяет: (1) оригинал и его аугментированные копии (по имени, только семейство 'aug'),
(2) файлы с одинаковым содержимым (md5), (3) почти-дубликаты по dHash.
Делить на train/val/test нужно по кластерам, иначе один и тот же снимок окажется и в обучении, и в оценке.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class DSU:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def near_pairs(bits: np.ndarray, max_thr: int = 12, chunk: int = 1024):
    """Все пары (i<j) с расстоянием Хэмминга <= max_thr. Возвращает индексы i, j и расстояния d."""
    f = bits.astype(np.float32)
    nb = f.shape[1]
    I, J, D = [], [], []
    for s in range(0, len(f), chunk):
        q = f[s:s + chunk]
        ham = nb - (q @ f.T + (1 - q) @ (1 - f).T)
        ii, jj = np.where(ham <= max_thr + 0.5)
        ii = ii + s
        m = ii < jj
        I.append(ii[m]); J.append(jj[m]); D.append(np.rint(ham[ii[m] - s, jj[m]]).astype(np.int32))
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)


def build_clusters(df: pd.DataFrame, ii, jj, dd, thr: int = 8) -> np.ndarray:
    df = df.reset_index(drop=True)
    dsu = DSU(len(df))
    for _, idx in df.groupby("md5").indices.items():
        for k in idx[1:]:
            dsu.union(idx[0], k)
    # имена объединяем только в семействе 'aug': в остальных именах (просто числа) совпадения случайны
    for _, idx in df[df.family == "aug"].groupby("group").indices.items():
        real = df[df.family == "aug"].index.values[idx]
        for k in real[1:]:
            dsu.union(real[0], k)
    for a, b in zip(ii[dd <= thr], jj[dd <= thr]):
        dsu.union(int(a), int(b))
    roots = np.array([dsu.find(i) for i in range(len(df))])
    return pd.factorize(roots)[0]


def make_splits(df: pd.DataFrame, cluster, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15,
                force_test=None):
    """Добавляет колонки cluster, eval_rep, role (train / val / test / drop).

    eval_rep: один настоящий (не синтетическая копия) снимок на кластер — только такие идут в val/test.
    Кластеры с разными классами внутри (шум разметки или ложное слияние) получают role='drop'.
    force_test: кластеры, которые обязательно попадают в test (например, «чистая» часть выданного теста).
    """
    df = df.reset_index(drop=True).copy()
    df["cluster"] = np.asarray(cluster)
    df["eval_rep"] = False
    real = df[df.aug.isin(["ORIG", "UNK", "NA"])]
    df.loc[real.drop_duplicates("cluster").index, "eval_rep"] = True

    ncls = df.groupby("cluster").cls.nunique()
    conflict = set(ncls[ncls > 1].index)
    cl_cls = df.groupby("cluster").cls.agg(lambda s: s.value_counts().index[0])
    has_rep = df.groupby("cluster").eval_rep.any()

    ev = cl_cls[has_rep & ~cl_cls.index.isin(conflict)]
    role = {c: "train" for c in cl_cls.index}
    for c in conflict:
        role[c] = "drop"

    n_total = len(ev)
    forced = [c for c in (force_test or []) if c in ev.index]
    for c in forced:
        role[c] = "test"
    pool = ev.drop(forced)

    n_test_extra = max(0, round(test_frac * n_total) - len(forced))
    if n_test_extra > 0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_test_extra, random_state=seed)
        rest_i, test_i = next(sss.split(pool.index.values, pool.values))
        for c in pool.index.values[test_i]:
            role[c] = "test"
        pool = pool.iloc[rest_i]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=round(val_frac * n_total), random_state=seed)
    _, val_i = next(sss2.split(pool.index.values, pool.values))
    for c in pool.index.values[val_i]:
        role[c] = "val"

    df["role"] = df.cluster.map(role)
    return df


def assert_clean(df: pd.DataFrame):
    """Проверка отсутствия утечки: кластер целиком в одной части, в val/test ровно один эталонный кадр на кластер."""
    used = df[df.role.isin(["train", "val", "test"])]
    assert (used.groupby("cluster").role.nunique() == 1).all(), "Один кластер попал в несколько частей!"
    ev = df[df.role.isin(["val", "test"])]
    reps = ev[ev.eval_rep]
    assert not reps.duplicated("cluster").any(), "В val/test по несколько эталонных кадров на кластер"
    assert set(ev.cluster) == set(reps.cluster), "В val/test есть кластер без эталонного кадра"
