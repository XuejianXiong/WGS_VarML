#!/usr/bin/env python3
"""
05_QC_feature.py

Generate an HTML QC-check report for ML-ready VEP feature matrices.

Report sections (in order):
1. Numeric Features Summary
2. Histograms (numeric features)
3. CLNSIG Label Distribution (if applicable)
4. Impact Frequencies
5. Top 15 Consequences
6. Missing Values Summary
7. Random Sample of Variants

Notes
-----
- Empty columns (e.g. SIFT, PolyPhen) are automatically skipped.
- CLNSIG tables/figures are skipped if all labels are -1.
- Numeric histograms use different colors; Impact and Consequences use gradient colors.

Usage:
    python3 src/05_QC_feature.py <features_csv> [output.html]

Example:
    python3 src/05_QC_feature.py data/processed/clinvar.vep.features.csv
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

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
setup_logger()


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a compact HTML table (not full width)."""
    return (
        df.head(max_rows)
        .to_html(
            border=0,
            classes="table",
            index=True,
            escape=False,
        )
    )


# ------------------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------------------
def plot_numeric_histogram(
    series: pd.Series,
    title: str,
    xlabel: str,
    clip_99pct: bool = False,
    note: str | None = None,
    color: str | None = None
) -> str:
    """Safely plot a numeric histogram if data exists."""
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
def numeric_summary_section(df: pd.DataFrame) -> str:
    numeric_cols = ["sift", "polyphen", "protein_position", "distance"]
    present = [c for c in numeric_cols if c in df.columns]

    summary = df[present].describe().T
    return f"""
    <h2>Numeric Features Summary</h2>
    {html_table(summary)}
    """


def histogram_section(df: pd.DataFrame) -> str:
    html = "<h2>Numeric Feature Distributions</h2>"

    # Distinct colors per numeric histogram
    colors = {"sift": "skyblue", "polyphen": "salmon", "protein_position": "lightgreen", "distance": "violet"}

    if "sift" in df.columns and df["sift"].notna().any():
        html += plot_numeric_histogram(df["sift"], "Histogram: SIFT", "SIFT score", color=colors["sift"])

    if "polyphen" in df.columns and df["polyphen"].notna().any():
        html += plot_numeric_histogram(df["polyphen"], "Histogram: PolyPhen", "PolyPhen score", color=colors["polyphen"])

    if "protein_position" in df.columns and df["protein_position"].notna().any():
        html += plot_numeric_histogram(
            df["protein_position"],
            "Histogram: Protein Position",
            "Amino acid position",
            color=colors["protein_position"]
        )

    if "distance" in df.columns and df["distance"].notna().any():
        html += plot_numeric_histogram(
            df["distance"],
            "Histogram: Distance to Nearest Gene",
            "Distance (bp)",
            clip_99pct=False,
            #note="Histogram clipped at 99th percentile for visualization; underlying distances unchanged.",
            color=colors["distance"]
        )

    return html


def clnsig_section(df: pd.DataFrame) -> str:
    CLNSIG_LABEL_MAP = {
          -1: "unknown",
          0: "benign",
          1: "pathogenic"
    }

    if "clnsig_label" not in df.columns:
        return ""

    # Skip if all unknown
    if df["clnsig_label"].nunique() == 1 and df["clnsig_label"].iloc[0] == -1:
        logger.info("CLNSIG labels all unknown; skipping CLNSIG section")
        return ""

    # Map numeric labels to human-readable
    df_labels = df["clnsig_label"].map(CLNSIG_LABEL_MAP)

    counts = df_labels.value_counts().rename("count")
    counts.index.name = None    

    fig, ax = plt.subplots(figsize=(4, 3))
    counts.plot(kind="bar", ax=ax, edgecolor="black", color=["orange", "green", "red"])
    ax.set_title("CLNSIG Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")

    img = fig_to_base64(fig)

    return f"""
    <h2>CLNSIG Label Distribution</h2>
    {html_table(counts.to_frame())}
    <img src="data:image/png;base64,{img}">
    """


def impact_section(df: pd.DataFrame) -> str:
    impact_cols = [c for c in df.columns if c.startswith("impact_")]
    if not impact_cols:
        return ""

    counts = df[impact_cols].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(5, 3))

    # Gradient colors
    colors = cm.viridis(np.linspace(0.2, 0.8, len(counts)))
    counts.plot(kind="bar", ax=ax, edgecolor="black", color=colors)
    ax.set_title("Impact Frequencies")
    ax.set_ylabel("Count")

    img = fig_to_base64(fig)

    return f"""
    <h2>Impact Frequencies</h2>
    {html_table(counts.to_frame(name="count"))}
    <img src="data:image/png;base64,{img}">
    """


def consequence_section(df: pd.DataFrame, n_cons: int = 10) -> str:
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

    # Horizontal bar gradient
    colors = cm.viridis(np.linspace(0.2, 0.8, len(counts)))
    counts.plot(kind="barh", ax=ax, edgecolor="black", color=colors)
    ax.set_title(f"Top {n_cons} Consequences")
    ax.invert_yaxis()
    ax.set_xlabel("Count")

    img = fig_to_base64(fig)

    return f"""
    <h2>Top {n_cons} Consequences</h2>
    {html_table(counts.to_frame(name="count"))}
    <img src="data:image/png;base64,{img}">
    """


def missing_values_section(df: pd.DataFrame) -> str:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    return f"""
    <h2>Missing Values</h2>
    {html_table(missing.to_frame(name="missing_count"))}
    """


def sample_section(df: pd.DataFrame, n_sample: int = 5) -> str:
    sample = df.sample(n=n_sample, random_state=42)
    return f"""
    <h2>Random Sample ({n_sample} variants)</h2>
    {html_table(sample, max_rows=n_sample)}
    """


# ------------------------------------------------------------------------------
# Main report generator
# ------------------------------------------------------------------------------
def generate_report(df: pd.DataFrame, n_cons: int, n_samples: int, output_html: str) -> None:
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

    html += numeric_summary_section(df)
    html += histogram_section(df)
    html += clnsig_section(df)
    html += impact_section(df)
    html += consequence_section(df, n_cons)
    html += missing_values_section(df)
    html += sample_section(df, n_samples)

    html += "</body></html>"

    Path(output_html).write_text(html)
    logger.info("QC report generation completed")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        logger.error("Usage: python 05_QC_feature.py <features.csv> [output.html]")
        sys.exit(1)

    features_csv = Path(sys.argv[1])
    output_html = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else features_csv.with_name(features_csv.stem + "_report.html")
    )

    logger.info(f"Reading feature matrix: {features_csv}")
    df = pd.read_csv(features_csv, low_memory=False)
    logger.info(f"Feature matrix shape: {df.shape}")

    n_cons = 15     # number of top consequences
    n_samples = 10  # number of random samples
    generate_report(df, n_cons, n_samples, str(output_html))


if __name__ == "__main__":
    main()
