import numpy as np
import cv2
from PIL import Image
from brainmri.serving import (to_gray_array, fit_reference, nn_distance, region_from_cam, traffic_light,
                              render_overlay, format_report, RU_NAMES)
from brainmri.data.preprocess import preprocess, preprocess_gray, load_gray


def test_to_gray_matches_training_loader(tmp_path):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)
    p = tmp_path / "a.png"; Image.fromarray(rgb).save(p)
    assert np.array_equal(to_gray_array(rgb), load_gray(p))
    assert np.array_equal(to_gray_array(np.dstack([rgb, np.full((40, 50), 255, np.uint8)])), load_gray(p))   # RGBA
    g = rng.integers(0, 255, (30, 30), dtype=np.uint8)
    assert np.array_equal(to_gray_array(g), g)
    assert to_gray_array(rng.random((20, 20)).astype(np.float32)).dtype == np.uint8


def test_preprocess_gray_equals_preprocess(tmp_path):
    img = np.zeros((300, 280), np.uint8); cv2.ellipse(img, (140, 150), (100, 120), 0, 0, 360, 120, -1)
    p = tmp_path / "h.png"; Image.fromarray(img).save(p)
    assert np.array_equal(preprocess(p), preprocess_gray(load_gray(p)))


def test_reference_distance():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(200, 32)) + 5.0                      # большая общая компонента
    mean, bank = fit_reference(feats)
    d_in = nn_distance(feats[:10], mean, bank)
    d_out = nn_distance(rng.normal(size=(10, 32)) + 5.0, mean, bank)
    assert d_in.max() < 0.01 and d_out.min() > d_in.max() + 0.05


def test_region_from_cam():
    cam = np.zeros((224, 224), np.float32); cam[60:100, 120:170] = 1.0; cam[10:14, 10:14] = 1.0   # очаг и мелкое пятно
    mask, box, frac = region_from_cam(cam)
    assert box == (120, 60, 170, 100) and abs(frac - 40 * 50 / 224 ** 2) < 1e-6
    assert region_from_cam(np.zeros((10, 10)))[1] is None


def test_traffic_light_levels():
    assert traffic_light(0.95, 0.01, 0.89, 0.05)["level"] == "green"
    assert traffic_light(0.80, 0.01, 0.89, 0.05)["level"] == "yellow"
    assert traffic_light(0.95, 0.20, 0.89, 0.05)["level"] == "yellow"
    assert traffic_light(0.80, 0.20, 0.89, 0.05)["level"] == "red"
    assert traffic_light(0.95, None, 0.89, None)["level"] == "green"        # без опорного набора


def test_render_and_report():
    img = np.random.default_rng(0).random((224, 224)).astype(np.float32)
    cam = np.zeros((224, 224), np.float32); cam[80:140, 80:140] = 1
    mask, box, frac = region_from_cam(cam)
    out = render_overlay(img, cam, mask)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    txt = format_report(classes, np.array([.9, .05, .03, .02]), 0, 0.92, 0.01, traffic_light(0.92, 0.01, 0.89, 0.05), frac)
    assert "Глиома" in txt and "не является медицинским изделием" in txt and "не сегментация" in txt
    txt2 = format_report(classes, np.array([.02, .02, .9, .06]), 2, 0.92, None, traffic_light(0.92, None, 0.89, None), 0.0)
    assert "Нет опухоли" in txt2


class _FakePredictor:
    """Подставная модель: проверяем всю цепочку интерфейса без torch и gradio."""
    t_conf, t_dist = 0.894, 0.05
    def __init__(self, pred=0, conf=0.93, dist=0.01):
        self.pred, self.conf, self.dist = pred, conf, dist
    def predict(self, gray):
        assert gray.ndim == 2 and gray.dtype == np.uint8
        img = np.random.default_rng(0).random((224, 224)).astype(np.float32)
        cam = np.zeros((224, 224), np.float32); cam[70:130, 90:150] = 1
        probs = np.full(4, (1 - self.conf) / 3); probs[self.pred] = self.conf
        return dict(img=img, cam=cam, probs=probs, pred=self.pred, conf=self.conf, dist=self.dist,
                    classes=["glioma", "meningioma", "notumor", "pituitary"])


def test_analyze_image_flow():
    import sys, importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("gradio_app", pathlib.Path(__file__).resolve().parents[1] / "app" / "gradio_app.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)          # gradio при импорте не нужен
    rgb = np.random.default_rng(1).integers(0, 255, (120, 100, 3), dtype=np.uint8)

    proc, cam, label, text = mod.analyze_image(_FakePredictor(), rgb)
    assert proc.shape == (224, 224, 3) and cam.shape == (224, 224, 3)
    assert set(label) == {"Глиома", "Менингиома", "Нет опухоли", "Опухоль гипофиза"} and abs(sum(label.values()) - 1) < 1e-6
    assert "🟢" in text and "Глиома" in text

    _, _, _, t_red = mod.analyze_image(_FakePredictor(conf=0.6, dist=0.3), rgb)
    assert "🔴" in t_red
    _, _, _, t_no = mod.analyze_image(_FakePredictor(pred=2), rgb)
    assert "Нет опухоли" in t_no and "не выделила" in t_no
    assert mod.analyze_image(_FakePredictor(), None)[3].startswith("Загрузите")
    assert mod.analyze_image(_FakePredictor(), np.zeros((0, 0)))[3].startswith("Не удалось")    # пустой файл не роняет приложение
