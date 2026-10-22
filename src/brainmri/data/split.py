"""Разбиение train/val БЕЗ утечки: все аугментированные копии одного снимка идут в одну часть."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def group_train_val_split(df: pd.DataFrame, val_fraction: float = 0.15, seed: int = 42):
    n_splits = max(2, round(1 / val_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(df, y=df["cls"], groups=df["group"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def assert_no_leak(a: pd.DataFrame, b: pd.DataFrame):
    inter = set(a["group"]) & set(b["group"])
    assert not inter, f"Утечка: {len(inter)} общих групп, пример: {list(inter)[:3]}"
