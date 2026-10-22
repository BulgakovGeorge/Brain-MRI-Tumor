import numpy as np
import cv2
from PIL import Image
from brainmri.data.preprocess import preprocess, augment, center_eval, head_bbox, load_gray


def _fake_head(w=300, h=345, text=True, mode="RGB", noise=False, path=None):
    """Синтетическая «голова»: эллипс с текстурой (одна и та же текстура в любом размере) и яркой оболочкой."""
    tex = np.random.default_rng(42).integers(40, 160, (64, 64)).astype(np.uint8)
    tex = cv2.resize(tex, (w, h), interpolation=cv2.INTER_CUBIC)
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 1, -1)
    img = tex * mask
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 230, max(2, w // 50))
    if text:                                        # «вшитый текст» в углу
        cv2.putText(img, "W 953 L 437", (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    if noise:
        m = np.random.default_rng(0).random(img.shape)
        img[m < 0.03] = 255; img[m > 0.97] = 0
    im = Image.fromarray(img).convert(mode)
    if path:
        im.save(path)
    return img, im


def test_bbox_ignores_corner_text():
    img, _ = _fake_head()
    x0, y0, x1, y1 = head_bbox(img)
    assert x0 > 40 and y1 < img.shape[0] - 20          # рамка не растянулась на текст в левом нижнем углу


def test_bbox_ignores_salt_pepper():
    img, _ = _fake_head(noise=True)
    x0, y0, x1, y1 = head_bbox(img)
    assert (x1 - x0) < 0.85 * img.shape[1]


def test_preprocess_modes_and_shape(tmp_path):
    for i, mode in enumerate(["RGB", "L", "RGBA", "P"]):
        for (w, h) in [(300, 345), (512, 512), (224, 224), (630, 630)]:
            _, im = _fake_head(w, h, mode=mode)
            p = tmp_path / f"{mode}_{w}.png"; im.save(p)
            out = preprocess(p)
            assert out.shape == (256, 256) and out.dtype == np.uint8
            assert out.max() > 200 and out.mean() > 20  # голова не потерялась


def test_format_independence(tmp_path):
    """Один и тот же «снимок» в разных размерах и пропорциях после обработки почти совпадает."""
    outs = []
    for (w, h) in [(300, 345), (512, 600), (224, 260)]:
        _, im = _fake_head(w, h, text=False)
        p = tmp_path / f"s{w}.png"; im.save(p); outs.append(preprocess(p).astype(np.float32))
    for o in outs[1:]:
        assert np.abs(o - outs[0]).mean() < 12


def test_augment_range_and_determinism():
    base = (np.random.default_rng(1).random((256, 256)) * 255).astype(np.uint8)
    a = augment(base, np.random.default_rng(5)); b = augment(base, np.random.default_rng(5))
    assert a.shape == (224, 224) and a.dtype == np.float32 and 0 <= a.min() and a.max() <= 1
    assert np.array_equal(a, b)
    assert center_eval(base).shape == (224, 224)


def _textured_head(w, h, rx, ry):
    tex = np.random.default_rng(42).integers(40, 160, (64, 64)).astype(np.uint8)
    tex = cv2.resize(tex, (w, h), interpolation=cv2.INTER_CUBIC)
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (rx, ry), 0, 0, 360, 1, -1)
    img = tex * mask
    cv2.ellipse(img, (w // 2, h // 2), (rx, ry), 0, 0, 360, 230, 4)
    return img


def test_corner_blob_removed(tmp_path):
    """Яркое пятно (как вшитый текст) внутри квадрата кропа, но вдали от головы, должно исчезнуть."""
    img = _textured_head(340, 340, 125, 125)
    cv2.rectangle(img, (40, 280), (70, 295), 255, -1)
    p = tmp_path / "t.png"; Image.fromarray(img).save(p)
    out = preprocess(p)
    assert out.mean() > 20
    region = (slice(225, 246), slice(6, 36))
    assert out[region].max() == 0
    # контроль теста: без стирания пятно в этой области было бы видно
    from brainmri.data.preprocess import crop_square, head_bbox, normalize_intensity
    sq = cv2.resize(crop_square(img, head_bbox(img)), (256, 256), interpolation=cv2.INTER_AREA)
    assert normalize_intensity(sq)[region].max() > 150


def test_frame_lines_removed(tmp_path):
    """Рамка по краю кадра, слитая с головой, обнуляется."""
    img = _textured_head(300, 300, 149, 149)
    cv2.rectangle(img, (0, 0), (299, 299), 200, 1)
    p = tmp_path / "f.png"; Image.fromarray(img).save(p)
    out = preprocess(p)
    b = 5
    assert out[:b].max() == 0 and out[-b:].max() == 0 and out[:, :b].max() == 0 and out[:, -b:].max() == 0
    assert out.mean() > 20
