from pathlib import Path

import pandas as pd

from src.migration.pipeline import GROUPS, standardize_measures
from src.migration.validation import compare_csv

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "Project_1" / "Starrating" / "alldata_2025jul.csv"
EXPECTED = ROOT / "data" / "Project_1" / "SAS Output CSV" / "STD_DATA_2025JUL_ANALYSIS.csv"


def test_standardization_preserves_hospitals_and_expected_measure_count():
    frame = pd.read_csv(INPUT, dtype={"PROVIDER_ID": "string"})
    result, included = standardize_measures(frame, 100)
    assert len(result) == 4566
    assert len(included) == 46
    assert result["PROVIDER_ID"].equals(frame["PROVIDER_ID"])
    assert result["Total_m_cnt"].max() <= len(included)


def test_groups_cover_the_sas_measure_domains():
    assert set(GROUPS) == {
        "OUTCOME_MORTALITY", "OUTCOME_SAFETY", "OUTCOME_READMISSION",
        "PTEXP", "PROCESS",
    }
    assert sum(map(len, GROUPS.values())) == 46


def test_validation_detects_a_real_difference(tmp_path):
    expected = tmp_path / "expected.csv"
    actual = tmp_path / "actual.csv"
    pd.DataFrame({"PROVIDER_ID": ["1"], "value": [1.0]}).to_csv(expected, index=False)
    pd.DataFrame({"PROVIDER_ID": ["1"], "value": [2.0]}).to_csv(actual, index=False)
    comparison = compare_csv(expected, actual)
    assert not comparison.passed
    assert "numeric mismatch" in comparison.messages[0]
