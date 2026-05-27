"""Download QANTA/Protobowl data from HuggingFace and save as JSONL."""
import json
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DATASET = "community-datasets/qanta"
CONFIG = "mode=full,char_skip=25"


def download_split(split_name: str, out_path: Path) -> int:
    print(f"Downloading {split_name}...")
    ds = load_dataset(DATASET, CONFIG, split=split_name)
    with open(out_path, "w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
    print(f"  Saved {len(ds)} rows to {out_path}")
    return len(ds)


if __name__ == "__main__":
    splits = {
        "guesstrain": DATA_DIR / "guesstrain.jsonl",
        "buzztrain": DATA_DIR / "buzztrain.jsonl",
        "guessdev": DATA_DIR / "guessdev.jsonl",
        "buzzdev": DATA_DIR / "buzzdev.jsonl",
        "guesstest": DATA_DIR / "guesstest.jsonl",
        "buzztest": DATA_DIR / "buzztest.jsonl",
    }

    total = 0
    for split, path in splits.items():
        if path.exists():
            lines = sum(1 for _ in open(path))
            print(f"  {split} already downloaded ({lines} rows), skipping.")
            total += lines
        else:
            total += download_split(split, path)

    print(f"\nTotal rows downloaded: {total}")
    size_mb = sum(p.stat().st_size for p in DATA_DIR.glob("*.jsonl")) / 1e6
    print(f"Data directory size: {size_mb:.1f} MB")
