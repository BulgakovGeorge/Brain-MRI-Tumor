"""Анализ достоверности результатов (без torch): источники данных, точность по расстоянию до train, доля Grad-CAM на фоне."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
import cv2


def source_of(family, pattern) -> str:
    """Грубое определение источника снимка по шаблону имени файла (цифры в именах заменены на #)."""
    if not isinstance(pattern, str):
        pattern = ""
    if family == "aug":
        return "A: больничный набор (с аугментациями)"
    if pattern.startswith(("Tr-", "Te-")):
        return "B: Tr-/Te-"
    if pattern == "#":
        return "C: просто числа"
    if re.match(r"^[A-Za-z]+#? \(#\)$", pattern):
        return "D: буквы и (число)"
    return "E: прочее"


def stratified_accuracy(dist, correct, edges=None, n_bins: int = 4) -> pd.DataFrame:
    """Точность по корзинам расстояния до ближайшего снимка из train. edges=None -> квантили."""
    dist = np.asarray(dist, float)
    correct = np.asarray(correct, bool)
    if edges is None:
        edges = np.quantile(dist, np.linspace(0, 1, n_bins + 1))
    edges = np.asarray(edges, float)
    b = np.digitize(dist, edges[1:-1])
    rows = []
    for k in range(len(edges) - 1):
        m = b == k
        if m.sum() == 0:
            continue
        rows.append({"диапазон": f"{edges[k]:.3g} - {edges[k + 1]:.3g}", "n": int(m.sum()),
                     "acc": float(correct[m].mean()), "ошибок": int((~correct[m]).sum())})
    return pd.DataFrame(rows)


def bg_fraction(cam: np.ndarray, img01: np.ndarray, thr: float = 0.08, dilate: int = 9) -> float:
    """Доля «массы» карты внимания вне головы (фон и края кадра). Большая доля = подозрение на шорткат."""
    head = cv2.dilate((img01 > thr).astype(np.uint8), np.ones((dilate, dilate), np.uint8)).astype(bool)
    tot = float(cam.sum())
    return float(cam[~head].sum() / tot) if tot > 0 else 0.0


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Доверительный интервал Уилсона для доли k/n (хорошо работает при малых n и долях у 0 или 1)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def coverage_table(conf, correct, fracs=(0, 0.02, 0.05, 0.1, 0.2, 0.3)) -> pd.DataFrame:
    """Что будет, если отказываться от самых неуверенных предсказаний (передавать врачу).
    Порог берётся по квантилям уверенности, поэтому таблица не зависит от абсолютной калибровки."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, bool)
    rows = []
    for f in fracs:
        thr = -np.inf if f == 0 else float(np.quantile(conf, f))
        keep = conf >= thr
        rows.append({"отказ от доли": f, "порог": None if f == 0 else round(thr, 4), "покрытие": float(keep.mean()),
                     "acc": float(correct[keep].mean()), "ошибок осталось": int((~correct[keep]).sum())})
    return pd.DataFrame(rows)


def ece(conf, correct, bins: int = 10) -> float:
    """Ожидаемая ошибка калибровки: насколько уверенность отличается от реальной точности."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, bool)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)
