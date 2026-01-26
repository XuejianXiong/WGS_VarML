#!/usr/bin/env python3
"""
05_QC_feature.py

Generate an HTML QC-check report for ML-ready VEP feature matrices.

Report sections (in order):
1. Numeric feature summary
2. Numeric feature histograms (conditional)
3. CLNSIG label distribution (conditional)
4. Impact frequencies
5. Top consequences
6. Missing value summary
7. Random sample of variants

Notes:
- Production-safe for multi-million rows
- Skip empty / degenerate features automatically
- Clear logging
- Support a YAML config file

Usage:
    python3 src/05_QC_feature.py <features_csv> <output.html> <config.yaml>

Example:
    python3 src/05_QC_feature.py data/processed/clinvar.vep.features.csv results/qc_report.html config/config.yaml
"""

from __future__ import annotations

import sys
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from logzero import logger, setup_logger

# Shared config utilities
from utils.config import load_config, resolve

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
setup_logger()


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib Figure into a base64-encoded PNG string.

    This allows the figure to be embedded directly into HTML using:
        <img src="data:image/png;base64,...">
    """
    # Create an in-memory binary buffer
    buffer = BytesIO()

    # Save the figure into the buffer as a PNG image
    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        dpi=120,
    )
    plt.close(fig)

    # Convert the buffer contents to base64-encoded text
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return image_base64


def html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    Render a compact HTML table.
    """
    html = (
        df.head(max_rows)
        .to_html(
            border=0,
            classes="table",
            index=True,
            escape=False,
        )
    )

    return html

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
    Plot a numeric histogram if data exists.

    Clipping (if enabled) affects visualization only,
    never the underlying data.
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

    img = fig_to_base64(fig)

    html = f"""
    <h4>{title}</h4>
    <img src="data:image/png;base64,{img}">
    """

    if note:
        html += f"<p><em>{note}</em></p>"

    return html


# ------------------------------------------------------------------------------
# Report sections
# ------------------------------------------------------------------------------
def numeric_summary_section(df: pd.DataFrame, numeric_cols: list[str]) -> str:
    """
    Summary statistics for numeric features.
    """
    present = [c for c in numeric_cols if c in df.columns]
    if not present:
        return ""

    summary = df[present].describe().T

    html = f"""
            <h2>Numeric Features Summary</h2>
            {html_table(summary)}
            """

    return html


def histogram_section(
    df: pd.DataFrame,
    numeric_cols: list[str],
    clip_distance: bool,
    clip_note: str | None,
) -> str:
    """
    Plotting histograms for numeric features.
    """
    html = "<h2>Numeric Feature Distributions</h2>"

    colors = {
        "sift": "skyblue",
        "polyphen": "salmon",
        "protein_position": "lightgreen",
        "distance": "violet",
    }

    for col in numeric_cols:
        if col not in df.columns or not df[col].notna().any():
            continue

        html += plot_numeric_histogram(
            df[col],
            title=f"Histogram: {col}",
            xlabel=col.replace("_", " ").title(),
            clip_99pct=(col == "distance" and clip_distance),
            note=clip_note if col == "distance" else None,
            color=colors.get(col),
        )

    return html


def clnsig_section(df: pd.DataFrame) -> str:
    """
    CLNSIG label distribution (skipped if all unknown).
    """
    CLNSIG_LABEL_MAP = {
        -1: "unknown",
        0: "benign",
        1: "pathogenic",
    }

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

    img = fig_to_base64(fig)

    html = f"""
            <h2>CLNSIG Label Distribution</h2>
            {html_table(counts.to_frame())}
            <img src="data:image/png;base64,{img}">
            """
            
    return html


def impact_section(df: pd.DataFrame) -> str:
    """
    VEP impact frequency summary.
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

    img = fig_to_base64(fig)

    html = f"""
            <h2>Impact Frequencies</h2>
            {html_table(counts.to_frame(name="count"))}
            <img src="data:image/png;base64,{img}">
            """
    return html


def consequence_section(df: pd.DataFrame, n_cons: int) -> str:
    """
    Top N VEP consequences.
    """
    cons_cols = [c for c in df.columns if c.startswith("cons_")]
    if not cons_cols:
        return ""

    counts = (
        df[cons_cols]
        .sum()
        .sort_values(ascending=False)
        .head(n_cons)
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = cm.viridis(np.linspace(0.2, 0.8, len(counts)))
    counts.plot(kind="barh", ax=ax, edgecolor="black", color=colors)
    ax.set_title(f"Top {n_cons} Consequences")
    ax.invert_yaxis()
    ax.set_xlabel("Count")

    img = fig_to_base64(fig)

    html = f"""
            <h2>Top {n_cons} Consequences</h2>
            {html_table(counts.to_frame(name="count"))}
            <img src="data:image/png;base64,{img}">
            """
    return html


def missing_values_section(df: pd.DataFrame) -> str:
    """
    Missing value summary.
    """
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    html = f"""
            <h2>Missing Values</h2>
            {html_table(missing.to_frame(name="missing_count"))}
            """

    return html


def sample_section(df: pd.DataFrame, n_sample: int) -> str:
    """
    Random sample of variants.
    """
    sample = df.sample(n=n_sample, random_state=42)

    html = f"""
            <h2>Random Sample ({n_sample} variants)</h2>
            {html_table(sample, max_rows=n_sample)}
            """

    return html


# ------------------------------------------------------------------------------
# Report generator
# ------------------------------------------------------------------------------
def generate_report(df: pd.DataFrame, config: dict, output_html: str) -> None:
    """
    Generate QC report using config-driven parameters.
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

    logger.info(f"Generating HTML QC report: {output_html}")

    html = """
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
    """

    html += numeric_summary_section(df, numeric_cols)
    html += histogram_section(df, numeric_cols, clip_distance, clip_note)
    html += clnsig_section(df)
    html += impact_section(df)
    html += consequence_section(df, n_cons)
    html += missing_values_section(df)
    html += sample_section(df, n_samples)

    html += "</body></html>"

    Path(output_html).write_text(html)
    logger.info("QC report generation completed")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        logger.error(
            "Usage: python 05_QC_feature.py <features.csv> [output.html] [config.yaml]"
        )
        sys.exit(1)

    features_csv = Path(sys.argv[1])

    output_html = (
        Path(sys.argv[2])
        if len(sys.argv) >= 3 and not sys.argv[2].endswith(".yaml")
        else features_csv.with_name(features_csv.stem + "_report.html")
    )

    config_path = Path(sys.argv[3]) if len(sys.argv) >= 4 else None
    config = load_config(config_path)

    logger.info(f"Reading feature matrix: {features_csv}")
    suffix = features_csv.suffix.lower()
    
    if suffix == ".csv":
        df = pd.read_csv(features_csv, low_memory=False)
    elif suffix == ".parquet":
        df = pd.read_parquet(features_csv)
    else:
        raise ValueError(f"Unsupported feature format: {suffix}")
    
    logger.info(f"Feature matrix shape: {df.shape}")

    generate_report(df, config, str(output_html))


if __name__ == "__main__":
    main()
