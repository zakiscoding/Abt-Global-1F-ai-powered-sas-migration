from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Comparison:
    path: str
    passed: bool
    messages: tuple[str, ...]


def compare_csv(expected_path: Path, actual_path: Path, tolerance: float = 1e-6) -> Comparison:
    expected = pd.read_csv(expected_path, dtype={"PROVIDER_ID": "string"})
    actual = pd.read_csv(actual_path, dtype={"PROVIDER_ID": "string"})
    messages: list[str] = []
    if list(expected.columns) != list(actual.columns):
        messages.append("column order or names differ")
    if len(expected) != len(actual):
        messages.append(f"row count differs: expected {len(expected)}, actual {len(actual)}")
    shared = [column for column in expected.columns if column in actual.columns]
    for column in shared:
        left, right = expected[column], actual[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(left.to_numpy(), right.to_numpy(), equal_nan=True, atol=tolerance, rtol=tolerance):
                difference = (left - right).abs().max()
                messages.append(f"{column}: numeric mismatch, max absolute difference {difference:g}")
        elif not left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)):
            messages.append(f"{column}: value mismatch")
    return Comparison(str(expected_path), not messages, tuple(messages))


def compare_directory(expected_dir: Path, actual_dir: Path, names: list[str]) -> list[Comparison]:
    return [
        compare_csv(expected_dir / name, actual_dir / name)
        for name in names
    ]
