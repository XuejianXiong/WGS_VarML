#!/usr/bin/env python3
"""
05_QC_feature.py

Generate an HTML quality-control (QC) report for ML-ready VEP feature matrices.
Used after 04_extract_features and before 06_split_clinvar to validate feature
distributions, label balance, and missingness.

Report sections (in order)
--------------------------
1. Numeric feature summary (mean, std, quartiles)
2. Numeric feature histograms (skipped if column empty; distance optionally clipped at 99th pct)
3. CLNSIG label distribution (skipped if all unknown)
4. Impact frequencies (one-hot impact_* columns)
5. Top N consequences (one-hot cons_* columns)
6. Missing value summary (columns with at least one NA)
7. Random sample of variants (fixed seed for reproducibility)

Config (config.yaml under "qc"): numeric_cols, top_consequences, random_samples,
distance_clip_pct, distance_clip_note. Precedence: CLI > YAML > defaults.

Usage
-----
    python3 src/05_QC_feature.py <features_csv_or_parquet> [--output REPORT.html] [--config CONFIG]

Example
-------
    python3 src/05_QC_feature.py data/processed/clinvar.vep.features.csv --output results/qc_report.html --config config/config.yaml
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from logzero import logger, setup_logger

from utils.config import load_config, resolve

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
setup_logger()

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
CLNSIG_LABEL_MAP = {-1: "unknown", 0: "benign", 1: "pathogenic"}

NUMERIC_HISTOGRAM_COLORS = {
    "sift": "skyblue",
    "polyphen": "salmon",
    "protein_position": "lightgreen",
    "distance": "violet",
}


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse CLI: input feature file (required), optional output path and config.
    """
    parser = argparse.ArgumentParser(
        description="Generate HTML QC report for VEP feature matrices."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to feature matrix (CSV or Parquet from 04_extract_features)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML path. If omitted, <input_stem>_report.html in same dir",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (qc.numeric_cols, top_consequences, etc.)",
    )
    return parser.parse_args()


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def fig_to_base64(fig: plt.Figure) -> str:
    """
    Encode a matplotlib Figure as a base64 PNG string for inline HTML embedding.

    Use in HTML as: <img src="data:image/png;base64,{return_value}">
    """
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_base64


def html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    Render a compact HTML table from a DataFrame (head only).
    """
    table_html = df.head(max_rows).to_html(
        border=0, classes="table", index=True, escape=False
    )
    return table_html


# ------------------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------------------
def plot_numeric_histogram(
    series: pd.Series,
    title: str,
    xlabel: str,
    clip_99pct: bool = False,
    note: str | None = None,
    color: str | None = None,
) -> str:
    """
    Build HTML snippet with a numeric histogram (or empty string if series has no data).

    Clipping (clip_99pct) affects only the plot; the underlying data is never modified.
    """
    values = series.dropna()
    if values.empty:
        logger.info(f"Skipping histogram (empty): {title}")
        return ""

    fig, ax = plt.subplots(figsize=(5, 3))
    if clip_99pct:
        xmax = values.quantile(0.99)
        plot_vals = values.clip(upper=xmax)
        ax.set_xlim(0, xmax)
    else:
        plot_vals = values

    ax.hist(plot_vals, bins=50, edgecolor="black", color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Variant count")

    img_b64 = fig_to_base64(fig)
    section_html = f"""
    <h4>{title}</h4>
    <img src="data:image/png;base64,{img_b64}">
    """
    if note:
        section_html += f"<p><em>{note}</em></p>"
    return section_html


# ------------------------------------------------------------------------------
# Report sections
# ------------------------------------------------------------------------------
def numeric_summary_section(df: pd.DataFrame, numeric_cols: list[str]) -> str:
    """
    Summary statistics (count, mean, std, quartiles) for numeric features.
    Returns empty string if none of the requested columns exist.
    """
    present = [c for c in numeric_cols if c in df.columns]
    if not present:
        return ""

    summary = df[present].describe().T
    section_html = f"""
            <h2>Numeric Features Summary</h2>
            {html_table(summary)}
            """
    return section_html


def histogram_section(
    df: pd.DataFrame,
    numeric_cols: list[str],
    clip_distance: bool,
    clip_note: str | None,
) -> str:
    """
    Histograms for numeric features; skips empty or all-NA columns.
    Optional 99th-percentile clipping for distance (visualization only).
    """
    section_html = "<h2>Numeric Feature Distributions</h2>"

    for col in numeric_cols:
        if col not in df.columns or not df[col].notna().any():
            continue
        section_html += plot_numeric_histogram(
            df[col],
            title=f"Histogram: {col}",
            xlabel=col.replace("_", " ").title(),
            clip_99pct=(col == "distance" and clip_distance),
            note=clip_note if col == "distance" else None,
            color=NUMERIC_HISTOGRAM_COLORS.get(col),
        )
    return section_html


def clnsig_section(df: pd.DataFrame) -> str:
    """
    CLNSIG label distribution (bar chart + table).
    Returns empty string if column missing or all labels are unknown.
    """
    if "clnsig_label" not in df.columns:
        return ""

    if df["clnsig_label"].nunique() == 1 and df["clnsig_label"].iloc[0] == -1:
        logger.info("CLNSIG labels all unknown; skipping CLNSIG section")
        return ""

    df_labels = df["clnsig_label"].map(CLNSIG_LABEL_MAP)
    counts = df_labels.value_counts().rename("count")

    fig, ax = plt.subplots(figsize=(4, 3))
    counts.plot(kind="bar", ax=ax, edgecolor="black", color=["orange", "green", "red"])
    ax.set_title("CLNSIG Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    img_b64 = fig_to_base64(fig)

    section_html = f"""
            <h2>CLNSIG Label Distribution</h2>
            {html_table(counts.to_frame())}
            <img src="data:image/png;base64,{img_b64}">
            """
    return section_html


def impact_section(df: pd.DataFrame) -> str:
    """
    VEP impact frequency summary (one-hot impact_* columns).
    Returns empty string if no impact_* columns exist.
    """
    impact_cols = [c for c in df.columns if c.startswith("impact_")]
    if not impact_cols:
        return ""

    counts = df[impact_cols].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = cm.viridis(np.linspace(0.2, 0.8, len(counts)))
    counts.plot(kind="bar", ax=ax, edgecolor="black", color=colors)
    ax.set_title("Impact Frequencies")
    ax.set_ylabel("Count")
    img_b64 = fig_to_base64(fig)

    section_html = f"""
            <h2>Impact Frequencies</h2>
            {html_table(counts.to_frame(name="count"))}
            <img src="data:image/png;base64,{img_b64}">
            """
    return section_html


def consequence_section(df: pd.DataFrame, n_cons: int) -> str:
    """
    Top N VEP consequences (one-hot cons_* columns).
    Returns empty string if no cons_* columns exist.
    """
    cons_cols = [c for c in df.columns if c.startswith("cons_")]
    if not cons_cols:
        return ""

    counts = df[cons_cols].sum().sort_values(ascending=False).head(n_cons)
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = cm.viridis(np.linspace(0.2, 0.8, len(counts)))
    counts.plot(kind="barh", ax=ax, edgecolor="black", color=colors)
    ax.set_title(f"Top {n_cons} Consequences")
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    img_b64 = fig_to_base64(fig)

    section_html = f"""
            <h2>Top {n_cons} Consequences</h2>
            {html_table(counts.to_frame(name="count"))}
            <img src="data:image/png;base64,{img_b64}">
            """
    return section_html


def missing_values_section(df: pd.DataFrame) -> str:
    """
    Per-column missing value counts (only columns with at least one NA).
    """
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    section_html = f"""
            <h2>Missing Values</h2>
            {html_table(missing.to_frame(name="missing_count"))}
            """
    return section_html


def sample_section(df: pd.DataFrame, n_sample: int) -> str:
    """
    Random sample of variants (fixed random_state=42 for reproducibility).
    If n_sample exceeds row count, uses all rows.
    """
    n_actual = min(n_sample, len(df))
    if n_actual == 0:
        return "<h2>Random Sample</h2><p>No variants to sample.</p>"
    sample = df.sample(n=n_actual, random_state=42)
    section_html = f"""
            <h2>Random Sample ({n_actual} variants)</h2>
            {html_table(sample, max_rows=n_actual)}
            """
    return section_html


# ------------------------------------------------------------------------------
# Report generator
# ------------------------------------------------------------------------------
def generate_report(df: pd.DataFrame, config: dict, output_path: str) -> None:
    """
    Build and write the full QC HTML report from the feature matrix and config.

    Config keys (under "qc"): numeric_cols, top_consequences, random_samples,
    distance_clip_pct, distance_clip_note. Empty DataFrame is supported;
    sections that need data will render empty or minimal content.
    """
    qc_cfg = config.get("qc", {})

    numeric_cols = resolve(
        None,
        qc_cfg.get("numeric_cols"),
        ["sift", "polyphen", "protein_position", "distance"],
    )
    n_cons = resolve(None, qc_cfg.get("top_consequences"), 10)
    n_samples = resolve(None, qc_cfg.get("random_samples"), 5)

    clip_pct = qc_cfg.get("distance_clip_pct")
    clip_distance = clip_pct is not None
    clip_note = qc_cfg.get("distance_clip_note")

    logger.info(f"Generating HTML QC report: {output_path}")

    if len(df) == 0:
        logger.warning("Feature matrix is empty; report will have minimal content")

    html_parts = [
        """
    <html>
    <head>
        <title>VEP Feature QC Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1, h2, h3 { margin-top: 40px; }
            .table { width: auto; border-collapse: collapse; }
            .table th, .table td {
                padding: 4px 8px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }
            img { margin-top: 10px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
    <h1>VEP Feature QC Report</h1>
    """,
        numeric_summary_section(df, numeric_cols),
        histogram_section(df, numeric_cols, clip_distance, clip_note),
        clnsig_section(df),
        impact_section(df),
        consequence_section(df, n_cons),
        missing_values_section(df),
        sample_section(df, n_samples),
        "</body></html>",
    ]
    html = "".join(html_parts)

    Path(output_path).write_text(html)
    logger.info("QC report generation completed")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def load_feature_matrix(path: Path) -> pd.DataFrame:
    """
    Load feature matrix from CSV or Parquet; format inferred from extension.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported feature format: {suffix}")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input feature file not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output is not None
        else input_path.with_name(input_path.stem + "_report.html")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(str(args.config) if args.config else None)

    logger.info(f"Reading feature matrix: {input_path}")
    df = load_feature_matrix(input_path)
    logger.info(f"Feature matrix shape: {df.shape}")

    generate_report(df, config, str(output_path))


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()



