"""Сбор манифеста датасета: путь, split, класс, исходный снимок (group), тип аугментации, источник."""
from __future__ import annotations
import hashlib, re
from pathlib import Path
import pandas as pd
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# M_861_HF_.jpg -> prefix=M, id=861, aug=HF ; M_862.jpg -> aug=ORIG (оригинал)
NAME_RE = re.compile(r"^(?P<prefix>[A-Za-z]+)_(?P<id>\d+)(?:_(?P<aug>[A-Za-z]{1,3}))?_*$")


def parse_name(stem: str):
    m = NAME_RE.match(stem)
    if not m:
        return None, None, None
    return m["prefix"], m["id"], (m["aug"] or "ORIG").upper()


def normalize_split(s: str) -> str:
    s = s.lower()
    if s.startswith("train"):
        return "train"
    if s.startswith("test"):
        return "test"
    if s.startswith("val"):
        return "val"
    return s


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def build_manifest(root: str | Path) -> pd.DataFrame:
    """Структура берётся с КОНЦА пути: .../<split>/<class>/<file>, поэтому лишние папки-обёртки не мешают."""
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or p.name.startswith("._") or "__MACOSX" in p.parts:
            continue
        rel = p.relative_to(root).parts
        if len(rel) >= 3:
            split, cls = normalize_split(rel[-3]), rel[-2].lower()
        elif len(rel) == 2:
            split, cls = "unknown", rel[-2].lower()
        else:
            split, cls = "unknown", "unknown"
        prefix, img_id, aug = parse_name(p.stem)
        group = f"{cls}/{prefix}_{img_id}" if prefix else f"{cls}/{p.stem}"
        row = dict(path=str(p), split=split, cls=cls, group=group,
                   aug=aug or "UNK",                               # UNK = имя не распознано
                   family="aug" if prefix else "other",            # aug = семейство с офлайн-аугментациями
                   pattern=re.sub(r"\d+", "#", p.stem),            # шаблон имени: Tr-gl_0010 -> Tr-gl_#
                   md5=md5(p), ok=True, w=None, h=None, mode=None)
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                row.update(w=im.width, h=im.height, mode=im.mode)
        except Exception:
            row["ok"] = False
        rows.append(row)
    return pd.DataFrame(rows)
