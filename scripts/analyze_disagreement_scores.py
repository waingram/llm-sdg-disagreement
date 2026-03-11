#!/usr/bin/env python3

# Description: Cross-reference Llama/Qwen disagreement rows with the original
# per-SDG Qwen probability CSVs and summarize whether disagreement cases fall in
# low teacher probability and teacher logit regions for their SDGs.

# Author: Bill Ingram <waingram@vt.edu>
# Date: Wed Mar 11 11:23:50 EDT 2026

# Usage: python scripts/analyze_disagreement_scores.py [--data-dir data --disagreement-csv data/model_disagreements.csv --output-dir data/analysis/disagreement_scores]

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SDG_PATTERN = re.compile(r"(\d+)")
RELEVANT = "Relevant"
NON_RELEVANT = "Non-Relevant"
REFERENCE_FILENAME_TEMPLATE = (
    "sdg{sdg}xsdg{sdg}_2023_{split}__scopus_sdg1_qwen_binary_bit_with_probs_v1.csv"
)
REFERENCE_SDGS = (1, 3, 7)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Join Llama/Qwen disagreement rows against the original per-SDG "
            "Qwen probability CSVs and summarize their p1/logit scores."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base data directory used for the default input and output paths.",
    )
    parser.add_argument(
        "--disagreement-csv",
        type=Path,
        default=None,
        help="Path to the disagreement CSV produced by 03_disagreement_analysis.ipynb.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where analysis CSV outputs should be written.",
    )
    return parser.parse_args()


def normalize_doi(value: object) -> str:
    """Normalize a DOI-like string for stable joins."""
    if value is None or pd.isna(value):
        return ""

    doi = str(value).strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi


def first_non_empty(values: Iterable[object]) -> str:
    """Return the first non-empty string-like value from an iterable."""
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def join_unique(values: Iterable[object], sep: str = "|") -> str:
    """Join unique non-empty values into a stable pipe-delimited string."""
    uniques = []
    seen = set()
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        uniques.append(text)
    return sep.join(sorted(uniques))


def extract_sdg_number(value: object) -> int:
    """Extract the numeric SDG id from labels like 'SDG1' or 'SDG 7'."""
    if value is None or pd.isna(value):
        raise ValueError("SDG value is missing.")

    match = SDG_PATTERN.search(str(value))
    if not match:
        raise ValueError(f"Could not parse SDG number from {value!r}.")
    return int(match.group(1))


def load_disagreement_rows(path: Path) -> pd.DataFrame:
    """Load and normalize the notebook-produced disagreement CSV."""
    df = pd.read_csv(path)
    required = {"DOI", "SDG", "Llama", "Qwen"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Disagreement CSV is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["doi_norm"] = df["DOI"].map(normalize_doi)
    df["sdg_number"] = df["SDG"].map(extract_sdg_number)
    df["disagreement_type"] = np.where(
        (df["Llama"] == RELEVANT) & (df["Qwen"] == NON_RELEVANT),
        "llama_relevant_qwen_non_relevant",
        np.where(
            (df["Llama"] == NON_RELEVANT) & (df["Qwen"] == RELEVANT),
            "qwen_relevant_llama_non_relevant",
            "other_disagreement",
        ),
    )
    return df


def infer_split_from_filename(path: Path) -> str:
    """Infer the train/test split from a reference CSV filename."""
    name = path.name.lower()
    if "_train__" in name:
        return "train"
    if "_test__" in name:
        return "test"
    return "unknown"


def load_reference_scores(data_dir: Path) -> pd.DataFrame:
    """Load and collapse the per-SDG Qwen probability CSVs."""
    frames = []
    for sdg_number in REFERENCE_SDGS:
        for split in ("train", "test"):
            path = data_dir / REFERENCE_FILENAME_TEMPLATE.format(
                sdg=sdg_number, split=split
            )
            if not path.exists():
                raise SystemExit(f"Reference score file does not exist: {path}")

            frame = pd.read_csv(path)
            required = {
                "row_id",
                "DOI",
                "SDG",
                "generated_text",
                "parsed_label",
                "logit_0",
                "logit_1",
                "teacher_logit",
                "p1",
            }
            missing = required - set(frame.columns)
            if missing:
                raise SystemExit(
                    f"Reference file {path.name} is missing required columns: "
                    f"{sorted(missing)}"
                )

            frame = frame.copy()
            frame["source_file"] = path.name
            frame["split"] = infer_split_from_filename(path)
            frame["doi_norm"] = frame["DOI"].map(normalize_doi)
            frame["sdg_number"] = frame["SDG"].map(extract_sdg_number)
            frames.append(frame)

    reference_df = pd.concat(frames, ignore_index=True)
    reference_df = reference_df[reference_df["doi_norm"].ne("")].copy()

    grouped = reference_df.groupby(["doi_norm", "sdg_number"], dropna=False)
    collapsed = (
        grouped[["p1", "teacher_logit", "logit_0", "logit_1"]].median().reset_index()
    )
    collapsed["DOI"] = grouped["DOI"].agg(first_non_empty).values
    collapsed["parsed_label"] = grouped["parsed_label"].agg(join_unique).values
    collapsed["generated_text"] = grouped["generated_text"].agg(join_unique).values
    collapsed["split"] = grouped["split"].agg(join_unique).values
    collapsed["source_files"] = grouped["source_file"].agg(join_unique).values
    collapsed["row_count"] = grouped.size().values
    return collapsed


def attach_reference_scores(
    disagreements: pd.DataFrame, reference_df: pd.DataFrame
) -> pd.DataFrame:
    """Join disagreement rows to the per-SDG Qwen probability records."""
    merged = disagreements.merge(
        reference_df,
        on=["doi_norm", "sdg_number"],
        how="left",
        suffixes=("", "_reference"),
    )
    merged["match_found"] = merged["p1"].notna() & merged["teacher_logit"].notna()
    merged["p1_score"] = merged["p1"]
    merged["teacher_logit_score"] = merged["teacher_logit"]
    return merged


def empirical_percentile(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Compute right-inclusive empirical percentiles against a reference series."""
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    clean_values = values.dropna()
    clean_reference = np.sort(reference.dropna().to_numpy())
    if clean_reference.size == 0 or clean_values.empty:
        return result
    result.loc[clean_values.index] = (
        np.searchsorted(clean_reference, clean_values.to_numpy(), side="right")
        / clean_reference.size
    )
    return result


def add_percentiles(
    row_level: pd.DataFrame, reference_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add within-SDG percentiles and baseline statistics."""
    output = row_level.copy()
    baseline_rows = []

    for sdg_number in sorted(output["sdg_number"].dropna().unique()):
        reference_sdg = reference_df[reference_df["sdg_number"] == sdg_number].copy()
        sdg_mask = output["sdg_number"] == sdg_number

        output.loc[sdg_mask, "p1_percentile"] = empirical_percentile(
            output.loc[sdg_mask, "p1_score"], reference_sdg["p1"]
        )
        output.loc[sdg_mask, "teacher_logit_percentile"] = empirical_percentile(
            output.loc[sdg_mask, "teacher_logit_score"], reference_sdg["teacher_logit"]
        )

        baseline_rows.append(
            {
                "SDG": f"SDG{sdg_number}",
                "sdg_number": sdg_number,
                "document_count": int(len(reference_sdg)),
                "p1_mean": float(reference_sdg["p1"].mean()),
                "p1_median": float(reference_sdg["p1"].median()),
                "p1_q25": float(reference_sdg["p1"].quantile(0.25)),
                "teacher_logit_mean": float(reference_sdg["teacher_logit"].mean()),
                "teacher_logit_median": float(reference_sdg["teacher_logit"].median()),
                "teacher_logit_q25": float(
                    reference_sdg["teacher_logit"].quantile(0.25)
                ),
            }
        )

    return output, pd.DataFrame(baseline_rows)


def summarize_groups(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate row-level disagreement score summaries."""
    rows = []
    for group_values, group_df in frame.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        matched_df = group_df[group_df["match_found"]].copy()
        row = dict(zip(group_cols, group_values))
        row.update(
            {
                "count": int(len(group_df)),
                "matched_count": int(group_df["match_found"].sum()),
                "unmatched_count": int((~group_df["match_found"]).sum()),
                "p1_mean": float(matched_df["p1_score"].mean()),
                "p1_median": float(matched_df["p1_score"].median()),
                "p1_percentile_mean": float(matched_df["p1_percentile"].mean()),
                "p1_percentile_median": float(matched_df["p1_percentile"].median()),
                "p1_share_below_q25": float(
                    (matched_df["p1_percentile"] <= 0.25).mean()
                ),
                "p1_share_below_median": float(
                    (matched_df["p1_percentile"] <= 0.50).mean()
                ),
                "teacher_logit_mean": float(matched_df["teacher_logit_score"].mean()),
                "teacher_logit_median": float(matched_df["teacher_logit_score"].median()),
                "teacher_logit_percentile_mean": float(
                    matched_df["teacher_logit_percentile"].mean()
                ),
                "teacher_logit_percentile_median": float(
                    matched_df["teacher_logit_percentile"].median()
                ),
                "teacher_logit_share_below_q25": float(
                    (matched_df["teacher_logit_percentile"] <= 0.25).mean()
                ),
                "teacher_logit_share_below_median": float(
                    (matched_df["teacher_logit_percentile"] <= 0.50).mean()
                ),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_outputs(
    output_dir: Path,
    row_level: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> None:
    """Persist all CSV outputs for the disagreement score analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    row_level.to_csv(output_dir / "disagreement_score_rows.csv", index=False)
    row_level.loc[~row_level["match_found"]].to_csv(
        output_dir / "disagreement_score_unmatched_rows.csv", index=False
    )
    baseline_df.to_csv(output_dir / "disagreement_score_baseline_by_sdg.csv", index=False)
    summarize_groups(row_level, ["SDG", "disagreement_type"]).to_csv(
        output_dir / "disagreement_score_summary.csv", index=False
    )
    summarize_groups(row_level, ["SDG", "disagreement_type", "split"]).to_csv(
        output_dir / "disagreement_score_summary_by_split.csv", index=False
    )


def print_console_summary(row_level: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    """Print a concise human-readable summary of the analysis."""
    total = len(row_level)
    matched = int(row_level["match_found"].sum())
    unmatched = total - matched

    print(f"Loaded {total} disagreement rows.")
    print(f"Matched {matched} rows to Qwen probability data; {unmatched} rows were unmatched.")

    summary_df = summarize_groups(row_level, ["SDG", "disagreement_type"])
    print("\nDisagreement score summary:")
    print(
        summary_df[
            [
                "SDG",
                "disagreement_type",
                "count",
                "matched_count",
                "p1_percentile_median",
                "teacher_logit_percentile_median",
            ]
        ].to_string(index=False)
    )

    print("\nReference baseline medians by SDG:")
    print(
        baseline_df[
            ["SDG", "document_count", "p1_median", "teacher_logit_median"]
        ].to_string(index=False)
    )


def main() -> None:
    """Run the disagreement-vs-score analysis against the per-SDG Qwen CSVs."""
    args = parse_args()
    data_dir = args.data_dir
    disagreement_csv = args.disagreement_csv or data_dir / "model_disagreements.csv"
    output_dir = args.output_dir or data_dir / "analysis" / "disagreement_scores"

    disagreements = load_disagreement_rows(disagreement_csv)
    reference_df = load_reference_scores(data_dir)
    row_level = attach_reference_scores(disagreements, reference_df)
    row_level, baseline_df = add_percentiles(row_level, reference_df)
    write_outputs(output_dir, row_level, baseline_df)
    print_console_summary(row_level, baseline_df)


if __name__ == "__main__":
    main()
