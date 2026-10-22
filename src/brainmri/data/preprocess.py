"""Единая предобработка снимков + аугментации.

Цель: убрать «шорткаты» по формату (размер, пропорции, режим, вшитый текст, яркость/контраст) и оставить анатомию.
Без torch: только numpy / PIL / OpenCV, чтобы всё проверялось без GPU.
"""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image

STORE = 256   # размер кэшированных стандартизованных снимков
CROP = 224    # размер, который идёт в сеть


def load_gray(path) -> np.ndarray:
    """Любой режим (RGB/L/RGBA/P) -> uint8 HxW."""
    with Image.open(path) as im:
        if im.mode != "L":
            im = im.convert("RGB").convert("L")
        return np.asarray(im, dtype=np.uint8)


def _fg_threshold(gray: np.ndarray) -> float:
    return max(8.0, 0.08 * float(np.percentile(gray, 99)))


def head_mask(gray: np.ndarray, open_k: int = 0):
    """Самая большая связная область ярче фона (голова). Возвращает (маска, рамка) или None.
    Мелкие пятна (вшитый текст, значки) — отдельные компоненты и в маску не попадают.
    open_k > 0 включает морфологическое открытие (по умолчанию выключено: рамку, слитую с головой, оно не убирает)."""
    h, w = gray.shape
    g = cv2.medianBlur(gray, 5)              # убирает шум «соль-перец»
    g = cv2.GaussianBlur(g, (0, 0), 2)
    mask = (g > _fg_threshold(g)).astype(np.uint8)
    if open_k:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[k]
    if area < 0.05 * h * w:
        return None
    return labels == k, (int(x), int(y), int(x + bw), int(y + bh))


def head_bbox(gray: np.ndarray):
    hm = head_mask(gray)
    return hm[1] if hm is not None else (0, 0, gray.shape[1], gray.shape[0])


def crop_square(gray: np.ndarray, box, margin: float = 0.04) -> np.ndarray:
    """Квадрат вокруг головы с чёрными полями (фон МРТ и так чёрный) — пропорции исходника больше не видны."""
    h, w = gray.shape
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (1 + 2 * margin) / 2
    a, b, c, d = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    pad = max(0, -a, -b, c - w, d - h)
    if pad:
        gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        a, b, c, d = a + pad, b + pad, c + pad, d + pad
    return gray[b:d, a:c]


def normalize_intensity(gray: np.ndarray, p=(1.0, 99.5)) -> np.ndarray:
    """Перцентильная нормировка по голове: убирает различия яркости и контраста между источниками."""
    mask = gray > _fg_threshold(gray)
    vals = gray[mask] if mask.sum() > 200 else gray.ravel()
    lo, hi = np.percentile(vals, p)
    out = np.clip((gray.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    return np.round(out * 255).astype(np.uint8)


def preprocess_gray(g: np.ndarray, size: int = STORE) -> np.ndarray:
    """Серый массив uint8 HxW -> uint8 size x size: всё кроме головы (текст, значки) стёрто, голова по центру в квадрате,
    рамка по краям обнулена, интенсивность нормирована."""
    hm = head_mask(g)
    if hm is not None:
        mask, box = hm
        if mask.sum() >= 0.15 * g.size:                       # маску применяем, только если голова найдена уверенно
            k = max(3, int(0.05 * max(g.shape)))
            keep = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
            g = g * keep
    else:
        box = (0, 0, g.shape[1], g.shape[0])
    sq = crop_square(g, box)
    interp = cv2.INTER_AREA if sq.shape[0] > size else cv2.INTER_CUBIC
    sq = cv2.resize(sq, (size, size), interpolation=interp)
    b = max(2, int(round(0.02 * size)))                       # рамки/линии по краю кадра
    sq[:b, :] = 0; sq[-b:, :] = 0; sq[:, :b] = 0; sq[:, -b:] = 0
    return normalize_intensity(sq)


def preprocess(path, size: int = STORE) -> np.ndarray:
    """Файл -> uint8 size x size (см. preprocess_gray)."""
    return preprocess_gray(load_gray(path), size)


def center_eval(img_u8: np.ndarray, out: int = CROP) -> np.ndarray:
    """Для val/test: просто уменьшаем без случайности. float32 [0,1]."""
    return cv2.resize(img_u8, (out, out), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def augment(img_u8: np.ndarray, rng: np.random.Generator, out: int = CROP) -> np.ndarray:
    """Аугментации для train. Геометрия + «порча качества», чтобы модель не опиралась на резкость, сжатие и шум.
    Вертикальные отражения не используем (голова вверх ногами в практике не встречается)."""
    s = img_u8.shape[0]
    # геометрия: поворот, масштаб, сдвиг, горизонтальное отражение
    ang = rng.uniform(-15, 15)
    sc = rng.uniform(0.9, 1.1) * out / s
    M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, sc)
    M[:, 2] += out / 2 - s / 2 + rng.uniform(-0.04, 0.04, 2) * out
    im = cv2.warpAffine(img_u8, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.5:
        im = im[:, ::-1]
    im = np.ascontiguousarray(im)

    # порча качества
    if rng.random() < 0.3:                                   # потеря разрешения
        f = rng.uniform(0.4, 0.8)
        small = cv2.resize(im, (int(out * f), int(out * f)), interpolation=cv2.INTER_AREA)
        im = cv2.resize(small, (out, out), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4:                                   # размытие
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.4:                                   # JPEG
        ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 96))])
        if ok:
            im = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    x = im.astype(np.float32) / 255.0
    if rng.random() < 0.5:                                   # гамма (яркость/контраст)
        x = x ** rng.uniform(0.7, 1.5)
    if rng.random() < 0.4:                                   # шум
        x = x + rng.normal(0, rng.uniform(0.005, 0.04), x.shape).astype(np.float32)
    if rng.random() < 0.5:                                   # чёрный прямоугольник в углу (как вшитый текст)
        sz = int(out * rng.uniform(0.05, 0.14))
        sz2 = int(out * rng.uniform(0.05, 0.14))
        top, left = rng.random() < 0.5, rng.random() < 0.5
        ys = slice(0, sz) if top else slice(out - sz, out)
        xs = slice(0, sz2) if left else slice(out - sz2, out)
        x[ys, xs] = 0
    return np.clip(x, 0, 1).astype(np.float32)
