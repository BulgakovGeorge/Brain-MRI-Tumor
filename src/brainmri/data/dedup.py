"""Поиск почти-дубликатов между выборками (dHash + расстояние Хэмминга)."""
from __future__ import annotations
import numpy as np
from PIL import Image


def dhash_bits(path, size: int = 16) -> np.ndarray:
    """size*size бит: сравнение соседних пикселей уменьшенной ч/б копии. Устойчив к яркости и сжатию."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).astype(np.uint8).ravel()


def nearest_cross(query: np.ndarray, ref: np.ndarray, chunk: int = 512):
    """Для каждой строки query: минимальное расстояние Хэмминга до ref и индекс ближайшей строки ref."""
    q = query.astype(np.float32)
    r = ref.astype(np.float32)
    n_bits = q.shape[1]
    dmin = np.empty(len(q), dtype=np.int32)
    amin = np.empty(len(q), dtype=np.int64)
    for i in range(0, len(q), chunk):
        qc = q[i:i + chunk]
        same = qc @ r.T + (1 - qc) @ (1 - r).T      # число совпавших бит
        ham = n_bits - same
        amin[i:i + chunk] = ham.argmin(1)
        dmin[i:i + chunk] = ham.min(1).round().astype(np.int32)
    return dmin, amin
