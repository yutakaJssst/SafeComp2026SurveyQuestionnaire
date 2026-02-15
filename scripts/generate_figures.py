#!/usr/bin/env python3
"""Generate publication-quality figures for:

    "Adoption of Safety Standards in the Japanese Automotive Industry"
    Matsuno, Ochiai, Kono — submitted to SafeComp 2026

Produces Fig. 1 (adoption status) as a grayscale horizontal stacked bar chart
suitable for monochrome printing (Springer LNCS).

Usage:
    python scripts/generate_figures.py                 # from repository root
    python scripts/generate_figures.py --input data/responses.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_adoption_data(csv_path: Path) -> dict:
    """Load adoption status counts from individual responses."""
    df = pd.read_csv(csv_path)
    data = {}
    for std, col in [("ISO 26262", "iso26262_adoption"),
                     ("SOTIF", "sotif_adoption"),
                     ("UL 4600", "ul4600_adoption")]:
        vc = df[col].value_counts()
        data[std] = {k: int(v) for k, v in vc.items()}
    return data, len(df)


def generate_fig_adoption(data: dict, output_path: Path, n_total: int = 30):
    """Generate Fig. 1: Standard adoption status (grayscale with hatching)."""
    standards = ["ISO 26262", "SOTIF", "UL 4600"]
    categories = ["In use", "Preparing", "Considering", "No plan"]

    # Grayscale colors + hatching for monochrome printing
    colors = ["#333333", "#777777", "#AAAAAA", "#DDDDDD"]
    hatches = ["", "//", "\\\\", "xx"]

    percentages = {}
    for std in standards:
        percentages[std] = []
        for cat in categories:
            count = data[std].get(cat, 0)
            percentages[std].append(count / n_total * 100)

    fig, ax = plt.subplots(figsize=(8, 3))
    y_pos = np.arange(len(standards))
    left = np.zeros(len(standards))

    bars_list = []
    for i, cat in enumerate(categories):
        values = [percentages[std][i] for std in standards]
        bars = ax.barh(y_pos, values, left=left, label=cat,
                       color=colors[i], hatch=hatches[i],
                       edgecolor="black", linewidth=0.5, height=0.5)
        bars_list.append(bars)
        left += values

    ax.set_yticks(y_pos)
    ax.set_yticklabels(standards, fontsize=11)
    ax.set_xlabel("Percentage (%)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

    # Print percentages for verification
    print(f"\nAdoption status percentages (n={n_total}):")
    for std in standards:
        print(f"  {std}:")
        for i, cat in enumerate(categories):
            print(f"    {cat}: {percentages[std][i]:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures from survey data"
    )
    parser.add_argument("--input", default="data/responses.csv",
                        help="Path to responses CSV (default: data/responses.csv)")
    parser.add_argument("--output-dir", default="figures",
                        help="Output directory for figures (default: figures)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, n = load_adoption_data(Path(args.input))
    generate_fig_adoption(data, output_dir / "fig_adoption.pdf", n_total=n)


if __name__ == "__main__":
    main()
