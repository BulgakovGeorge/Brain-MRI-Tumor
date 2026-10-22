import numpy as np
import cv2
from brainmri.data.stress import TRANSFORMS


def test_transforms_shape_dtype_determinism():
    img = (np.random.default_rng(0).random((256, 256)) * 255).astype(np.uint8)
    for name, tf in TRANSFORMS.items():
        a, b = tf(img), tf(img)
        assert a.shape == img.shape and a.dtype == np.uint8, name
        assert np.array_equal(a, b), name


def test_blur_and_lowres_reduce_sharpness_noise_adds_it():
    base = np.zeros((256, 256), np.uint8); cv2.circle(base, (128, 128), 80, 200, 6)
    lap = lambda x: cv2.Laplacian(x, cv2.CV_32F).var()
    assert lap(TRANSFORMS["размытие (sigma=2)"](base)) < lap(base)
    assert lap(TRANSFORMS["низкое разрешение (x0.4)"](base)) < lap(base)
    assert lap(TRANSFORMS["шум (sigma=0.06)"](base)) > lap(base)
