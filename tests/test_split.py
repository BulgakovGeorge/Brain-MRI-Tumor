import pandas as pd
from brainmri.data.split import group_train_val_split, assert_no_leak
from brainmri.data.manifest import parse_name


def test_parse_name():
    assert parse_name("M_861_HF_") == ("M", "861", "HF")
    assert parse_name("M_862") == ("M", "862", "ORIG")


def test_no_leak():
    rows = []
    for cls in ["a", "b"]:
        for i in range(40):
            for aug in ["ORIG", "HF", "VF", "RO"]:
                rows.append(dict(cls=cls, group=f"{cls}/X_{i}", aug=aug))
    df = pd.DataFrame(rows)
    tr, va = group_train_val_split(df, 0.2)
    assert_no_leak(tr, va)
    assert len(tr) + len(va) == len(df)
