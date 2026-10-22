"""Прототип: всё, что не требует torch (разбор входного изображения, «светофор доверия», область по карте внимания, отрисовка)."""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image

RU_NAMES = {"glioma": "Глиома", "meningioma": "Менингиома", "notumor": "Нет опухоли", "pituitary": "Опухоль гипофиза"}

DISCLAIMER = ("Исследовательский прототип, **не является медицинским изделием** и не ставит диагноз. "
              "Результат — вспомогательная подсказка для врача, окончательное решение принимает специалист.")


def to_gray_array(arr) -> np.ndarray:
    """Любое изображение (PIL, HxW, HxWx3, HxWx4, uint8/uint16/float) -> uint8 HxW тем же способом, что и при обучении."""
    if isinstance(arr, Image.Image):
        im = arr
    else:
        a = np.asarray(arr)
        if a.dtype != np.uint8:
            a = a.astype(np.float64)
            lo, hi = float(a.min()), float(a.max())
            a = np.round((a - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
        if a.ndim == 3 and a.shape[2] == 1:
            a = a[..., 0]
        im = Image.fromarray(a)
    if im.mode != "L":
        im = im.convert("RGB").convert("L")
    return np.asarray(im, dtype=np.uint8)


# ---------- «непохожесть на обучающие данные» ----------
def fit_reference(feats: np.ndarray):
    """Опорный набор: признаки обучающих снимков, центрированные и нормированные. Возвращает (mean, bank)."""
    f = np.asarray(feats, dtype=np.float32)
    mean = f.mean(0)
    bank = f - mean
    bank /= np.linalg.norm(bank, axis=1, keepdims=True) + 1e-9
    return mean, bank.astype(np.float16)


def nn_distance(feat: np.ndarray, mean: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Косинусное расстояние до ближайшего обучающего снимка. feat: (d,) или (n, d)."""
    f = np.atleast_2d(np.asarray(feat, dtype=np.float32)) - mean.astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
    sims = f @ bank.astype(np.float32).T
    return 1.0 - sims.max(1)


# ---------- область по карте внимания ----------
def region_from_cam(cam: np.ndarray, level: float = 0.5, min_area_frac: float = 0.01):
    """Связная область, где карта внимания >= level * максимума. Возвращает (маска uint8, рамка (x0, y0, x1, y1) или None, доля площади)."""
    cam = np.asarray(cam, dtype=np.float32)
    if cam.max() <= 0:
        return np.zeros(cam.shape, np.uint8), None, 0.0
    m = (cam >= level * cam.max()).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros(cam.shape, np.uint8), None, 0.0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[k, cv2.CC_STAT_AREA])
    if area < min_area_frac * cam.size:
        return np.zeros(cam.shape, np.uint8), None, 0.0
    x, y, w, h = (int(v) for v in stats[k, :4])
    return (labels == k).astype(np.uint8), (x, y, x + w, y + h), area / cam.size


# ---------- «светофор доверия» ----------
def traffic_light(conf: float, dist: float | None, t_conf: float, t_dist: float | None) -> dict:
    """Два независимых сигнала: низкая уверенность модели и нетипичность снимка для обучающих данных."""
    low_conf = conf < t_conf
    atypical = dist is not None and t_dist is not None and dist > t_dist
    if low_conf and atypical:
        return dict(level="red", title="Низкое доверие",
                    text="Модель не уверена, и снимок нетипичен для обучающих данных (другой источник, протокол или качество). "
                         "Результат ненадёжен, нужна оценка врача.")
    if low_conf:
        return dict(level="yellow", title="Сомнительный результат",
                    text="Уверенность модели ниже порога. Рекомендуется оценка врача.")
    if atypical:
        return dict(level="yellow", title="Нетипичный снимок",
                    text="Снимок отличается от обучающих данных (другой источник, протокол или качество). "
                         "Результат менее надёжен, рекомендуется оценка врача.")
    return dict(level="green", title="Типичный случай", text="Уверенность выше порога, снимок похож на обучающие данные.")


# ---------- отрисовка ----------
def render_overlay(img01: np.ndarray, cam: np.ndarray, mask: np.ndarray | None = None, alpha: float = 0.45) -> np.ndarray:
    """Серый снимок [0,1] + тепловая карта + контур области. Возвращает uint8 RGB."""
    base = np.repeat((np.clip(img01, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    out = cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)
    if mask is not None and mask.any():
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255), 2)
    return out


def to_rgb_uint8(img01: np.ndarray) -> np.ndarray:
    return np.repeat((np.clip(img01, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)


def format_report(classes, probs, pred: int, conf: float, dist: float | None, light: dict, region_frac: float | None) -> str:
    """Текст результата для интерфейса."""
    icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[light["level"]]
    name = RU_NAMES.get(classes[pred], classes[pred])
    lines = [f"## {name}", f"**{icon} {light['title']}.** {light['text']}", ""]
    lines.append(f"- Уверенность модели: **{conf:.2f}** (это оценка модели, а не вероятность диагноза)")
    if dist is not None:
        lines.append(f"- Непохожесть на обучающие данные: {dist:.3f}")
    if classes[pred] != "notumor" and region_frac:
        lines.append(f"- Область наибольшего внимания модели занимает ≈{region_frac * 100:.0f}% кадра (белый контур). "
                     "Это приблизительная область интереса по карте внимания, **не сегментация опухоли**.")
    elif classes[pred] == "notumor":
        lines.append("- Модель не выделила подозрительную область.")
    lines += ["", DISCLAIMER]
    return "\n".join(lines)
