from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

MEASURES = [
    "MORT_30_AMI", "MORT_30_CABG", "MORT_30_COPD", "MORT_30_HF",
    "MORT_30_PN", "MORT_30_STK", "PSI_4_SURG_COMP", "COMP_HIP_KNEE",
    "HAI_1", "HAI_2", "HAI_3", "HAI_4", "HAI_5", "HAI_6",
    "PSI_90_SAFETY", "EDAC_30_AMI", "EDAC_30_HF", "EDAC_30_PN",
    "OP_32", "READM_30_CABG", "READM_30_COPD", "READM_30_HIP_KNEE",
    "READM_30_HOSP_WIDE", "OP_35_ADM", "OP_35_ED", "OP_36",
    "H_COMP_1_STAR_RATING", "H_COMP_2_STAR_RATING",
    "H_COMP_3_STAR_RATING", "H_COMP_5_STAR_RATING",
    "H_COMP_6_STAR_RATING", "H_COMP_7_STAR_RATING",
    "H_GLOB_STAR_RATING", "H_INDI_STAR_RATING", "HCP_COVID_19",
    "IMM_3", "OP_10", "OP_13", "OP_18B", "OP_22", "OP_23", "OP_29",
    "OP_8", "PC_01", "SAFE_USE_OF_OPIOIDS", "SEP_1",
]

GROUPS = {
    "OUTCOME_MORTALITY": [
        "MORT_30_AMI", "MORT_30_CABG", "MORT_30_COPD", "MORT_30_HF",
        "MORT_30_PN", "MORT_30_STK", "PSI_4_SURG_COMP",
    ],
    "OUTCOME_SAFETY": [
        "COMP_HIP_KNEE", "HAI_1", "HAI_2", "HAI_3", "HAI_4", "HAI_5",
        "HAI_6", "PSI_90_SAFETY",
    ],
    "OUTCOME_READMISSION": [
        "EDAC_30_AMI", "EDAC_30_HF", "EDAC_30_PN", "OP_32",
        "READM_30_CABG", "READM_30_COPD", "READM_30_HIP_KNEE",
        "READM_30_HOSP_WIDE", "OP_35_ADM", "OP_35_ED", "OP_36",
    ],
    "PTEXP": [
        "H_COMP_1_STAR_RATING", "H_COMP_2_STAR_RATING",
        "H_COMP_3_STAR_RATING", "H_COMP_5_STAR_RATING",
        "H_COMP_6_STAR_RATING", "H_COMP_7_STAR_RATING",
        "H_GLOB_STAR_RATING", "H_INDI_STAR_RATING",
    ],
    "PROCESS": [
        "HCP_COVID_19", "IMM_3", "OP_10", "OP_13", "OP_18B", "OP_22",
        "OP_23", "OP_29", "OP_8", "PC_01", "SAFE_USE_OF_OPIOIDS", "SEP_1",
    ],
}

NEGATIVE_DIRECTION = set(MEASURES) - {
    "H_COMP_1_STAR_RATING", "H_COMP_2_STAR_RATING", "H_COMP_3_STAR_RATING",
    "H_COMP_5_STAR_RATING", "H_COMP_6_STAR_RATING", "H_COMP_7_STAR_RATING",
    "H_GLOB_STAR_RATING", "H_INDI_STAR_RATING", "HCP_COVID_19", "IMM_3",
    "OP_23", "OP_29", "SEP_1",
}


@dataclass(frozen=True)
class PipelineConfig:
    input_csv: Path
    output_dir: Path
    min_volume: int = 100


def _read_input(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"PROVIDER_ID": "string"})
    missing = {"PROVIDER_ID", *MEASURES} - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    return frame


def standardize_measures(frame: pd.DataFrame, min_volume: int) -> tuple[pd.DataFrame, list[str]]:
    included = [name for name in MEASURES if frame[name].notna().sum() > min_volume]
    result = frame.drop(columns=MEASURES).copy()
    result["Total_m_cnt"] = frame[included].notna().sum(axis=1)
    for name in included:
        # SAS PROC STANDARD uses sample standard deviation.
        result[f"std_{name}"] = (frame[name] - frame[name].mean()) / frame[name].std(ddof=1)
    for name in included:
        if name in NEGATIVE_DIRECTION:
            result[f"std_{name}"] = -result[f"std_{name}"]
    return result, included


def _group_score(frame: pd.DataFrame, measures: Iterable[str]) -> pd.DataFrame:
    standardized = [f"std_{name}" for name in measures if f"std_{name}" in frame]
    result = frame[["PROVIDER_ID", *standardized]].copy()
    result["total_cnt"] = result[standardized].notna().sum(axis=1)
    result["measure_wt"] = np.where(result["total_cnt"] > 0, 1 / result["total_cnt"], np.nan)
    result["score_before_std"] = result[standardized].sum(axis=1, min_count=1) * result["measure_wt"]
    mean = result["score_before_std"].mean()
    std = result["score_before_std"].std(ddof=1)
    result["Mean"] = mean
    result["StdDev"] = std
    result["grp_score"] = (result["score_before_std"] - mean) / std
    for index, name in enumerate(standardized, start=1):
        result[f"C{index}"] = result[name].notna().astype("int64")
    ordered = ["PROVIDER_ID", *standardized,
               *[f"C{i}" for i in range(1, len(standardized) + 1)],
               "total_cnt", "measure_wt", "score_before_std", "Mean", "StdDev",
               "grp_score"]
    return result[ordered]


def calculate_groups(analysis: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    outputs = {}
    for name, measures in GROUPS.items():
        result = _group_score(analysis, measures)
        outputs[name] = result
        result.to_csv(output_dir / f"{name}.csv", index=False)
    return outputs


def calculate_summary(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    scores = pd.DataFrame({"PROVIDER_ID": groups["OUTCOME_MORTALITY"]["PROVIDER_ID"]})
    aliases = {
        "OUTCOME_MORTALITY": "Std_Outcomes_Mortality_score",
        "OUTCOME_READMISSION": "Std_Outcomes_Readmission_score",
        "OUTCOME_SAFETY": "Std_Outcomes_Safety_score",
        "PTEXP": "Std_PatientExp_score",
        "PROCESS": "Std_Process_score",
    }
    for key, alias in aliases.items():
        scores[alias] = groups[key]["grp_score"].to_numpy()
    weights = {
        "Std_PatientExp_score": 0.22,
        "Std_Outcomes_Readmission_score": 0.22,
        "Std_Outcomes_Mortality_score": 0.22,
        "Std_Outcomes_Safety_score": 0.22,
        "Std_Process_score": 0.12,
    }
    available = scores[list(weights)].notna()
    denominator = sum(
        available[column].astype(float) * weight
        for column, weight in weights.items()
    )
    # Reallocation is row-wise, matching the SAS denominator logic.
    for column, weight in weights.items():
        scores[f"_weighted_{column}"] = scores[column] * weight / denominator
    scores["summary_score"] = scores.filter(regex="^_weighted_").sum(axis=1, min_count=1)
    return scores


def assign_stars(summary: pd.DataFrame, groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = summary.copy()
    counts = {}
    for key, group in groups.items():
        label = {
            "OUTCOME_MORTALITY": "Outcomes_Mortality_cnt",
            "OUTCOME_SAFETY": "Outcomes_safety_cnt",
            "OUTCOME_READMISSION": "Outcomes_Readmission_cnt",
            "PTEXP": "Patient_Experience_cnt",
            "PROCESS": "Process_cnt",
        }[key]
        counts[label] = group["total_cnt"].to_numpy()
    for label, values in counts.items():
        result[label] = values
    qualifying = pd.DataFrame(counts).ge(3)
    result["Total_measure_group_cnt"] = qualifying.sum(axis=1)
    result["MortSafe_group_cnt"] = (
        qualifying["Outcomes_Mortality_cnt"] + qualifying["Outcomes_safety_cnt"]
    )
    result["report_indicator"] = (
        (result["Total_measure_group_cnt"] >= 3)
        & (result["MortSafe_group_cnt"] >= 1)
    ).astype("int64")
    result["cnt_grp"] = result["Total_measure_group_cnt"].map({
        3: "1) # of groups=3",
        4: "2) # of groups=4",
        5: "3) # of groups=5",
    })
    result["star"] = np.nan
    eligible = result["report_indicator"].eq(1) & result["summary_score"].notna()
    for _, indices in result.loc[eligible].groupby("cnt_grp").groups.items():
        scores = result.loc[indices, "summary_score"].sort_values()
        buckets = pd.qcut(scores.rank(method="first"), q=5, labels=False, duplicates="drop")
        result.loc[buckets.index, "star"] = buckets.astype(float).to_numpy() + 1
    return result


def national_averages(stars: pd.DataFrame, groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    eligible = stars[stars["report_indicator"].eq(1)]
    row = {
        "Summary_Score_Nat": eligible["summary_score"].mean(),
        "Summary_Score_Nat_peer3": eligible.loc[eligible["Total_measure_group_cnt"].eq(3), "summary_score"].mean(),
        "Summary_Score_Nat_peer4": eligible.loc[eligible["Total_measure_group_cnt"].eq(4), "summary_score"].mean(),
        "Summary_Score_Nat_peer5": eligible.loc[eligible["Total_measure_group_cnt"].eq(5), "summary_score"].mean(),
    }
    for key, column in {
        "OUTCOME_MORTALITY": "Out_Mrt_Grp_Score_Nat",
        "OUTCOME_SAFETY": "Out_Sft_Grp_Score_Nat",
        "OUTCOME_READMISSION": "Out_Readm_grp_score_Nat",
        "PTEXP": "Pt_Exp_Grp_Score_Nat",
        "PROCESS": "Prc_of_Care_Grp_Score_Nat",
    }.items():
        row[column] = groups[key].loc[eligible.index, "grp_score"].mean()
    return pd.DataFrame([row])


def run_pipeline(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw = _read_input(config.input_csv)
    analysis, included = standardize_measures(raw, config.min_volume)
    analysis.to_csv(config.output_dir / "STD_DATA_2025JUL_ANALYSIS.csv", index=False)
    groups = calculate_groups(analysis, config.output_dir)
    summary = calculate_summary(groups)
    summary.to_csv(config.output_dir / "SUMMARY_SCORE.csv", index=False)
    stars = assign_stars(summary, groups)
    stars.to_csv(config.output_dir / "STAR_2025JUL.csv", index=False)
    national = national_averages(stars, groups)
    national.to_csv(config.output_dir / "NATIONAL_AVERAGE_2025JUL.csv", index=False)
    return {
        "analysis": analysis, "included_measures": included, **groups,
        "summary": summary, "stars": stars, "national": national,
    }
