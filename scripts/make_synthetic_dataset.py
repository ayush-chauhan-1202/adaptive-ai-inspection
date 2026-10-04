"""Generate a tiny synthetic MVTec-AD-shaped dataset for fast local/CI use.

The real MVTec AD "bottle" category is a few GB and requires accepting MVTec's
license on their site - not something CI (or a quick local smoke test) should
depend on downloading. This script writes a handful of small procedurally
generated images in the same directory layout `MVTecDataset` expects, so the
rest of the pipeline (feature extraction, memory bank, API, tests) can be
exercised end-to-end without it.

This is explicitly a smoke-test fixture, not a dataset substitute: scores and
metrics produced against it are meaningless as anomaly-detection results and
should never be compared against the real milestone 1-5 numbers.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

CATEGORY = "bottle"


def _make_image(rng: np.random.Generator, defect: bool) -> Image.Image:
    base = rng.normal(loc=150, scale=10, size=(64, 64, 3))

    if defect:
        # A bright square "scratch" in a random spot - visually and
        # statistically distinct from the plain noisy background.
        y, x = rng.integers(10, 40, size=2)
        base[y : y + 12, x : x + 12] = 255

    array = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def build(root: Path, seed: int = 42, n_train: int = 12, n_test_good: int = 6, n_test_bad: int = 6) -> None:
    rng = np.random.default_rng(seed)
    category_root = root / CATEGORY

    if category_root.exists():
        shutil.rmtree(category_root)

    train_good = category_root / "train" / "good"
    test_good = category_root / "test" / "good"
    test_bad = category_root / "test" / "scratch"
    mask_bad = category_root / "ground_truth" / "scratch"

    for directory in (train_good, test_good, test_bad, mask_bad):
        directory.mkdir(parents=True, exist_ok=True)

    for i in range(n_train):
        _make_image(rng, defect=False).save(train_good / f"{i:03d}.png")

    for i in range(n_test_good):
        _make_image(rng, defect=False).save(test_good / f"{i:03d}.png")

    for i in range(n_test_bad):
        _make_image(rng, defect=True).save(test_bad / f"{i:03d}.png")
        # A minimal mask so code paths that look for one don't break; content
        # doesn't matter for anything this project currently does with it.
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(
            mask_bad / f"{i:03d}_mask.png"
        )

    print(f"Synthetic dataset written to {category_root}")
    print(f"  train/good:   {n_train}")
    print(f"  test/good:    {n_test_good}")
    print(f"  test/scratch: {n_test_bad}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/synthetic_smoke_test"),
        help="Directory to write the synthetic category under.",
    )
    args = parser.parse_args()

    build(args.root)


if __name__ == "__main__":
    main()
