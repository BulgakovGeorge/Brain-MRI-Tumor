import numpy as np
from PIL import Image
from brainmri.data.manifest import build_manifest
from brainmri.data.dedup import dhash_bits, nearest_cross


def _img(p, seed=0):
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (32, 32), dtype=np.uint8)).save(p)


def test_wrapper_folder_is_ignored(tmp_path):
    base = tmp_path / "Some Wrapper"
    _img(base / "Train" / "glioma" / "M_1.jpg")
    _img(base / "Train" / "glioma" / "M_1_HF_.jpg", 1)
    _img(base / "Test" / "notumor" / "Te-no_0001.jpg", 2)
    df = build_manifest(tmp_path)
    assert set(df.split) == {"train", "test"}
    assert set(df.cls) == {"glioma", "notumor"}
    assert df[df.family == "aug"].group.nunique() == 1
    assert (df.family == "other").sum() == 1


def test_nearest_cross_finds_copy(tmp_path):
    _img(tmp_path / "a.png", 0); _img(tmp_path / "b.png", 1)
    A = np.stack([dhash_bits(tmp_path / "a.png"), dhash_bits(tmp_path / "b.png")])
    d, idx = nearest_cross(A[[1]], A)
    assert d[0] == 0 and idx[0] == 1


def test_clusters_and_splits():
    import pandas as pd
    from brainmri.data.cluster import build_clusters, make_splits, assert_clean
    rows = []
    for c in ["a", "b", "c", "d"]:
        for i in range(30):
            for aug in ["ORIG", "HF", "VF"]:
                rows.append(dict(cls=c, group=f"{c}/X_{i}", aug=aug, family="aug", md5=f"{c}{i}{aug}", split="train"))
    df = pd.DataFrame(rows)
    ii = jj = dd = np.array([], dtype=int)
    cl = build_clusters(df, ii, jj, dd)
    assert len(set(cl)) == 120
    out = make_splits(df, cl, seed=1)
    assert_clean(out)
    assert set(out.role) == {"train", "val", "test"}
    assert out[out.role == "test"].cls.nunique() == 4


def test_unknown_names_survive_csv_roundtrip(tmp_path):
    """Регрессия: строка 'NA' в csv читалась pandas как пропуск, и файлы с нераспознанными именами выпадали из оценки."""
    import pandas as pd
    base = tmp_path / "W"
    _img(base / "Train" / "glioma" / "Tr-gl_0001.jpg")
    df = build_manifest(tmp_path)
    p = tmp_path / "m.csv"; df.to_csv(p, index=False)
    back = pd.read_csv(p)
    assert back.aug.isin(["ORIG", "UNK", "NA"]).all()


def test_force_test_clusters():
    import pandas as pd
    from brainmri.data.cluster import build_clusters, make_splits, assert_clean
    rows = []
    for c in ["a", "b", "c", "d"]:
        for i in range(40):
            rows.append(dict(cls=c, group=f"{c}/X_{i}", aug="ORIG", family="aug", md5=f"{c}{i}", split="train"))
    df = pd.DataFrame(rows)
    e = np.array([], dtype=int)
    cl = build_clusters(df, e, e, e)
    forced = list(cl[:10])                      # первые 10 кластеров (все класса 'a') принудительно в test
    out = make_splits(df, cl, seed=3, force_test=forced)
    assert_clean(out)
    assert (out[out.cluster.isin(forced)].role == "test").all()
    n_total = out.eval_rep.sum()
    assert abs((out[out.role == "test"].eval_rep.sum()) - round(0.15 * n_total)) <= 1
