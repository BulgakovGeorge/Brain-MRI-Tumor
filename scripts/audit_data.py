"""Аудит датасета: python scripts/audit_data.py --root data/raw --out data/manifest.csv"""
import argparse, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brainmri.data.manifest import build_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default="data/manifest.csv")
    a = ap.parse_args()

    df = build_manifest(a.root)
    df.to_csv(a.out, index=False)
    print(f"Файлов: {len(df)}  | битых: {(~df.ok).sum()}")
    print("\nsplit x класс:\n", pd.crosstab(df.split, df.cls))
    print("\nтипы аугментаций (суффиксы в имени):\n", df.aug.value_counts())
    print("\nуникальных исходных снимков (group):", df.group.nunique())
    print("\nразмеры (топ-5):\n", df.groupby(["w", "h"]).size().sort_values(ascending=False).head())

    dup = df[df.duplicated("md5", keep=False)]
    print(f"\nточные дубликаты по md5: {len(dup)} файлов")

    if {"train", "test"} <= set(df.split):
        g_tr, g_te = set(df[df.split == "train"].group), set(df[df.split == "test"].group)
        print(f"\n!!! групп, одновременно в train и test: {len(g_tr & g_te)}")
        md_tr, md_te = set(df[df.split == "train"].md5), set(df[df.split == "test"].md5)
        print(f"!!! одинаковых файлов (md5) в train и test: {len(md_tr & md_te)}")


if __name__ == "__main__":
    main()
