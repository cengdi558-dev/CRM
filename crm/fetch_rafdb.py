"""Download RAF-DB from the HuggingFace mirror and lay it out as ImageFolder.

No Kaggle account, no registration form: the dataset is mirrored on
hf-mirror.com (the mainland mirror of the HuggingFace Hub).

    python -m crm.fetch_rafdb --out .rafdb --val-fraction 0.1

Produces::

    .rafdb/train/<class>/*.jpg
    .rafdb/val/<class>/*.jpg
    .rafdb/test/<class>/*.jpg

and prints the class order so it can be checked against the paper.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import urllib.request

BASE = "https://hf-mirror.com/datasets/elipaluma/RAF-DB_Kaggle/resolve/main/data/"
SHARDS = {"train": ["train-00000-of-00001.parquet"], "test": ["test-00000-of-00001.parquet"]}

#: RAF-DB's published order; used when the parquet carries no names
DEFAULT_CLASSES = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]


def download(url: str, path: str) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  cached {path}")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  GET {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, open(path, "wb") as fh:
        data = resp.read()
        fh.write(data)
    print(f"  saved {path} ({len(data) / 1048576:.1f} MB)")
    return path


def class_names(path: str) -> list[str]:
    """Read the ClassLabel names out of the parquet metadata, if present."""
    import pyarrow.parquet as pq

    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(b"huggingface")
    if not raw:
        return DEFAULT_CLASSES
    try:
        info = json.loads(raw.decode("utf-8"))
        names = info["info"]["features"]["label"]["names"]
        if isinstance(names, list) and len(names) == 7:
            return [str(n) for n in names]
    except Exception:
        pass
    return DEFAULT_CLASSES


def write_split(parquets: list[str], out_dir: str, names: list[str]) -> dict[str, int]:
    import pyarrow.parquet as pq
    from PIL import Image

    # Some Kaggle mirrors store the labels 1-based (1..7) while the metadata
    # lists seven names for 0..6. Detect that and shift, instead of silently
    # mapping every image onto the wrong class.
    rows = []
    seen: set[int] = set()
    for path in parquets:
        chunk = pq.read_table(path, columns=["image", "label"]).to_pylist()
        rows.extend(chunk)
        seen.update(int(r["label"]) for r in chunk)

    shift = 1 if (seen and min(seen) >= 1 and max(seen) == len(names)) else 0
    if shift:
        print(f"  note: labels are 1-based {sorted(seen)}; shifting by -1")

    counts: dict[str, int] = {}
    for row in rows:
        label = int(row["label"]) - shift
        name = names[label] if 0 <= label < len(names) else f"class{label}"
        folder = os.path.join(out_dir, name)
        os.makedirs(folder, exist_ok=True)
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        img.save(os.path.join(folder, f"{len(os.listdir(folder)):06d}.jpg"), quality=95)
        counts[name] = counts.get(name, 0) + 1
    return counts


def split_val(train_dir: str, val_dir: str, fraction: float, seed: int = 0) -> dict[str, int]:
    rng = random.Random(seed)
    moved = {}
    for cls in sorted(os.listdir(train_dir)):
        src = os.path.join(train_dir, cls)
        if not os.path.isdir(src):
            continue
        files = sorted(os.listdir(src))
        rng.shuffle(files)
        take = max(1, int(len(files) * fraction))
        dst = os.path.join(val_dir, cls)
        os.makedirs(dst, exist_ok=True)
        for f in files[:take]:
            os.replace(os.path.join(src, f), os.path.join(dst, f))
        moved[cls] = take
    return moved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".rafdb")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--cache", default=".rafdb/_parquet")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = None
    for split, files in SHARDS.items():
        final = os.path.join(args.out, split)
        marker = os.path.join(final, ".done")
        if os.path.exists(marker):
            print(f"{split}: already built, skipping")
            continue

        paths = []
        for f in files:
            p = download(BASE + f, os.path.join(args.cache, f))
            paths.append(p)
            if names is None:
                names = class_names(p)
        print(f"{split}: classes = {names}")
        counts = write_split(paths, final, names)
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(counts))
        print(f"{split}: {sum(counts.values())} images  {counts}")

    val_dir = os.path.join(args.out, "val")
    if not os.path.exists(os.path.join(val_dir, ".done")) and args.val_fraction > 0:
        moved = split_val(os.path.join(args.out, "train"), val_dir, args.val_fraction)
        open(os.path.join(val_dir, ".done"), "w", encoding="utf-8").write("ok")
        print(f"val: moved {sum(moved.values())} images  {moved}")

    print("done ->", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
