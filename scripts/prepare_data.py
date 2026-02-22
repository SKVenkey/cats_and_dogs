"""
Prepare the Cats vs Dogs dataset.
Downloads from Kaggle if kaggle CLI is configured, otherwise expects manual placement.

Usage:
    python scripts/prepare_data.py --data_dir data/raw

Dataset structure expected after running:
    data/raw/train/Cat/
    data/raw/train/Dog/
"""

import argparse
import os
import sys
import shutil
from pathlib import Path


def prepare_from_kaggle(data_dir: str):
    """Download and organise dataset from Kaggle."""
    import subprocess
    raw = Path(data_dir)
    raw.mkdir(parents=True, exist_ok=True)

    print("Downloading Cats vs Dogs dataset from Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "salader/dogs-vs-cats", "-p", str(raw), "--unzip"],
        check=True,
    )
    print("Download complete.")


def verify_structure(data_dir: str):
    """Check that the data directory has the expected structure."""
    train_cat = Path(data_dir) / "train" / "Cat"
    train_dog = Path(data_dir) / "train" / "Dog"

    issues = []
    if not train_cat.exists():
        issues.append(f"Missing: {train_cat}")
    if not train_dog.exists():
        issues.append(f"Missing: {train_dog}")

    if issues:
        print("Dataset structure issues found:")
        for i in issues:
            print(f"  {i}")
        print("\nExpected structure:")
        print("  data/raw/train/Cat/  (containing .jpg images)")
        print("  data/raw/train/Dog/  (containing .jpg images)")
        sys.exit(1)
    else:
        n_cats = len(list(train_cat.glob("*.jpg"))) + len(list(train_cat.glob("*.png")))
        n_dogs = len(list(train_dog.glob("*.jpg"))) + len(list(train_dog.glob("*.png")))
        print(f"Dataset OK — Cats: {n_cats}, Dogs: {n_dogs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--download", action="store_true", help="Download from Kaggle")
    args = parser.parse_args()

    if args.download:
        prepare_from_kaggle(args.data_dir)

    verify_structure(args.data_dir)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
