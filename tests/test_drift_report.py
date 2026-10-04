"""Tests for the Milestone 9 drift report script."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# scripts/ is a collection of standalone CLI entry points, not an installed
# package (consistent with how run_patchcore.py etc. have always been run
# directly) - so it's added to sys.path here rather than via pyproject's
# pythonpath config, which is scoped to src/ for the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_drift_report import run_drift_report


def _write_scores(path: Path, mean: float, rng: np.random.Generator) -> None:
    pd.DataFrame({"score": rng.normal(mean, 1.0, 200)}).to_csv(path, index=False)


def test_no_drift_when_distributions_match(tmp_path: Path):
    rng = np.random.default_rng(0)
    reference_csv = tmp_path / "reference.csv"
    current_csv = tmp_path / "current.csv"

    _write_scores(reference_csv, mean=0.0, rng=rng)
    _write_scores(current_csv, mean=0.0, rng=rng)

    drift_detected = run_drift_report(reference_csv, current_csv, tmp_path / "report.html")

    assert drift_detected is False
    assert (tmp_path / "report.html").exists()


def test_drift_detected_on_distribution_shift(tmp_path: Path):
    rng = np.random.default_rng(0)
    reference_csv = tmp_path / "reference.csv"
    current_csv = tmp_path / "current.csv"

    _write_scores(reference_csv, mean=0.0, rng=rng)
    _write_scores(current_csv, mean=3.0, rng=rng)

    drift_detected = run_drift_report(reference_csv, current_csv, tmp_path / "report.html")

    assert drift_detected is True
