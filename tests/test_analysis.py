import numpy as np
from brainmri.analysis import source_of, stratified_accuracy, bg_fraction


def test_source_of():
    assert source_of("aug", "G_#_BR_").startswith("A")
    assert source_of("other", "Tr-gl_#").startswith("B") and source_of("other", "Te-no_#").startswith("B")
    assert source_of("other", "#").startswith("C")
    assert source_of("other", "gg (#)").startswith("D") and source_of("other", "m# (#)").startswith("D")
    assert source_of("other", "p (#)").startswith("D")
    assert source_of("other", float("nan")).startswith("E")


def test_stratified_accuracy():
    d = np.arange(100.0); ok = d < 50                       # близкие снимки угаданы, далёкие нет
    t = stratified_accuracy(d, ok, edges=[0, 25, 50, 75, 100.1])
    assert t.n.tolist() == [25, 25, 25, 25] and t.acc.tolist() == [1.0, 1.0, 0.0, 0.0]
    q = stratified_accuracy(d, ok)                           # квантили
    assert q.n.sum() == 100 and len(q) == 4


def test_bg_fraction():
    img = np.zeros((224, 224), np.float32); img[60:160, 60:160] = 0.5
    cam_in = np.zeros_like(img); cam_in[80:140, 80:140] = 1
    cam_out = np.zeros_like(img); cam_out[0:20, 0:20] = 1
    assert bg_fraction(cam_in, img) < 0.01 and bg_fraction(cam_out, img) > 0.99


def test_wilson_coverage_ece():
    from brainmri.analysis import wilson_ci, coverage_table, ece
    lo, hi = wilson_ci(0, 100); assert lo == 0.0 and 0.03 < hi < 0.04
    lo, hi = wilson_ci(50, 100); assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    rng = np.random.default_rng(0)
    conf = rng.random(1000); correct = rng.random(1000) < conf        # идеально откалиброванная уверенность
    t = coverage_table(conf, correct)
    assert t.покрытие.is_monotonic_decreasing and t.acc.iloc[-1] > t.acc.iloc[0]
    assert ece(conf, correct) < 0.06
    assert ece(np.ones(1000), rng.random(1000) < 0.5) > 0.4
