"""Стресс-тест: ухудшаем качество снимков и смотрим, насколько падают метрики. Вход и выход: uint8 HxW."""
from __future__ import annotations
import numpy as np
import cv2


def _rng(img):
    return np.random.default_rng(int(img.astype(np.uint64).sum() % (2 ** 32)))   # детерминированно по содержимому


def identity(img): return img

def blur(img, sigma=2.0):
    return cv2.GaussianBlur(img, (0, 0), sigma)

def noise(img, sigma=0.06):
    x = img.astype(np.float32) / 255 + _rng(img).normal(0, sigma, img.shape).astype(np.float32)
    return np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)

def lowres(img, f=0.4):
    h, w = img.shape
    small = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def jpeg(img, q=20):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE) if ok else img

def gamma(img, g=1.6):
    return np.round(((img.astype(np.float32) / 255) ** g) * 255).astype(np.uint8)

def rotate(img, deg=10.0):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

def zoom(img, s=1.2):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


TRANSFORMS = {
    "без искажений": identity,
    "размытие (sigma=2)": blur,
    "шум (sigma=0.06)": noise,
    "низкое разрешение (x0.4)": lowres,
    "JPEG (q=20)": jpeg,
    "темнее (гамма 1.6)": gamma,
    "ярче (гамма 0.6)": lambda im: gamma(im, 0.6),
    "поворот 10 градусов": rotate,
    "крупнее (x1.2)": zoom,
}
