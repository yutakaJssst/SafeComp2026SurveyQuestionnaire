#!/usr/bin/env python3
"""Reproduce all statistical analyses reported in:

    "Adoption of Safety Standards in the Japanese Automotive Industry"
    Matsuno, Ochiai, Kono — submitted to SafeComp 2026

This script reads individual-level survey data (data/responses.csv) and
reproduces every statistical test, effect size, and multiple-comparison
correction reported in the paper.

Usage:
    python scripts/analyze.py                  # from repository root
    python scripts/analyze.py --input data/responses.csv
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================
# Statistical test functions
# ============================================================================

def _wilcoxon_mode(n_nonzero: int) -> str:
    """Use exact p-values for small samples (n <= 25)."""
    return "exact" if n_nonzero <= 25 else "auto"


def wilcoxon_paired(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    """Wilcoxon signed-rank test for paired samples with effect size.

    Effect size r = |Z| / sqrt(N) where N = n_nonzero.
    Only reported when n_nonzero >= 10 (unreliable for small samples).
    """
    mask = x.notna() & y.notna()
    xv, yv = x[mask].values, y[mask].values
    n_pair = len(xv)
    if n_pair == 0:
        return {"n_pair": 0, "n_nonzero": 0, "W": float("nan"),
                "p": float("nan"), "r": float("nan")}
    diff = xv - yv
    nz = diff[diff != 0]
    n_nonzero = len(nz)
    if n_nonzero == 0:
        return {"n_pair": n_pair, "n_nonzero": 0, "W": 0.0,
                "p": 1.0, "r": 0.0}
    mode = _wilcoxon_mode(n_nonzero)
    res = stats.wilcoxon(nz, zero_method="wilcox", correction=False,
                         alternative="two-sided", mode=mode)
    if n_nonzero < 10:
        r = float("nan")
    else:
        mean_w = n_nonzero * (n_nonzero + 1) / 4
        var_w = n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24
        z = (res.statistic - mean_w) / math.sqrt(var_w)
        r = min(abs(z) / math.sqrt(n_nonzero), 1.0)
    return {"n_pair": int(n_pair), "n_nonzero": int(n_nonzero),
            "W": float(res.statistic), "p": float(res.pvalue),
            "r": float(r)}


def mann_whitney_u(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    """Mann-Whitney U test with tie-corrected effect size."""
    xv, yv = x.dropna().values, y.dropna().values
    n1, n2 = len(xv), len(yv)
    if n1 == 0 or n2 == 0:
        return {"n1": n1, "n2": n2, "U": float("nan"),
                "p": float("nan"), "r": float("nan")}
    res = stats.mannwhitneyu(xv, yv, alternative="two-sided", method="auto")
    combined = np.concatenate([xv, yv])
    _, counts = np.unique(combined, return_counts=True)
    N = n1 + n2
    mean_u = n1 * n2 / 2
    tie_correction = sum(int(c) ** 3 - int(c) for c in counts if c > 1)
    var_u = n1 * n2 / 12 * ((N + 1) - tie_correction / (N * (N - 1))) if N > 1 else n1 * n2 * (N + 1) / 12
    z = (res.statistic - mean_u) / math.sqrt(var_u) if var_u > 0 else 0.0
    r = min(abs(z) / math.sqrt(N), 1.0)
    return {"n1": int(n1), "n2": int(n2), "U": float(res.statistic),
            "p": float(res.pvalue), "r": float(r)}


def one_sample_wilcoxon(x: pd.Series, median: float = 3.0) -> Dict[str, float]:
    """One-sample Wilcoxon signed-rank test against a hypothetical median."""
    xv = x.dropna().values
    n = len(xv)
    if n == 0:
        return {"n": 0, "n_nonzero": 0, "W": float("nan"),
                "p": float("nan"), "r": float("nan")}
    diff = xv - median
    nz = diff[diff != 0]
    n_nonzero = len(nz)
    if n_nonzero == 0:
        return {"n": n, "n_nonzero": 0, "W": 0.0, "p": 1.0, "r": 0.0}
    mode = _wilcoxon_mode(n_nonzero)
    res = stats.wilcoxon(nz, zero_method="wilcox", correction=False,
                         alternative="two-sided", mode=mode)
    if n_nonzero < 10:
        r = float("nan")
    else:
        mean_w = n_nonzero * (n_nonzero + 1) / 4
        var_w = n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24
        z = (res.statistic - mean_w) / math.sqrt(var_w)
        r = min(abs(z) / math.sqrt(n_nonzero), 1.0)
    return {"n": int(n), "n_nonzero": int(n_nonzero),
            "W": float(res.statistic), "p": float(res.pvalue),
            "r": float(r)}


def friedman_test(df_sub: pd.DataFrame) -> Tuple[int, float, float]:
    """Friedman test across columns (listwise deletion of NaN)."""
    clean = df_sub.dropna()
    n = len(clean)
    if n == 0:
        return 0, float("nan"), float("nan")
    stat, p = stats.friedmanchisquare(*[clean[col] for col in clean.columns])
    return n, float(stat), float(p)


def holm_correction(pvals: List[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down correction."""
    arr = np.array(pvals, dtype=float)
    adj = np.full_like(arr, np.nan)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return adj.tolist()
    vals = arr[mask]
    m = len(vals)
    order = np.argsort(vals)
    prev = 0.0
    out = np.empty(m)
    for rank, idx in enumerate(order):
        adj_val = (m - rank) * vals[idx]
        prev = max(prev, adj_val)
        out[idx] = min(prev, 1.0)
    adj[mask] = out
    return adj.tolist()


def desc(s: pd.Series) -> Tuple[int, float, float]:
    """Return n, mean, SD for a numeric series (dropping NaN)."""
    v = s.dropna()
    n = len(v)
    if n == 0:
        return 0, float("nan"), float("nan")
    if n == 1:
        return 1, float(v.mean()), float("nan")
    return n, float(v.mean()), float(v.std(ddof=1))


# ============================================================================
# Formatting helpers
# ============================================================================

def fp(p):
    if math.isnan(p): return "---"
    return f"{p:.6f}" if p < 0.001 else f"{p:.4f}"

def fr(r):
    if math.isnan(r): return "---"
    return f"{r:.3f}"

def ff(v, d=2):
    if math.isnan(v): return "---"
    return f"{v:.{d}f}"

def sig(p_holm):
    if math.isnan(p_holm): return ""
    if p_holm < 0.001: return "***"
    if p_holm < 0.01: return "**"
    if p_holm < 0.05: return "*"
    return ""


# ============================================================================
# Column name definitions
# ============================================================================

DRIVER_COLS = ["technical_safety", "2nd_party_accountability",
               "3rd_party_accountability", "contract_requirements",
               "industry_conformity"]
DRIVER_LABELS = {
    "technical_safety": "Technical safety",
    "2nd_party_accountability": "2nd-party accountability",
    "3rd_party_accountability": "3rd-party accountability",
    "contract_requirements": "Contract requirements",
    "industry_conformity": "Industry conformity",
}

BC_COLS = ["safety_improvement", "operational_burden", "cost_effectiveness"]
BC_LABELS = {
    "safety_improvement": "Safety improvement",
    "operational_burden": "Operational burden",
    "cost_effectiveness": "Cost-effectiveness",
}

ROLE_COLS = ["upper_requirements", "validation", "detailed_design",
             "implementation", "integration"]
ROLE_LABELS = {
    "upper_requirements": "Upper requirements definition",
    "validation": "Requirements validation",
    "detailed_design": "Detailed design",
    "implementation": "Implementation & unit verification",
    "integration": "Integration & system verification",
}


# ============================================================================
# Main analysis
# ============================================================================

def run_analysis(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Reproducibility Analysis Results")
    lines.append("")
    lines.append(f"N = {len(df)} respondents")
    lines.append("")

    # --- Profile ---
    lines.append("## Respondent Profile")
    lines.append("")
    for col, label in [("company_type", "Company type"),
                       ("role", "Role"),
                       ("aspice_status", "A-SPICE status"),
                       ("iso26262_experience", "ISO 26262 experience")]:
        vc = df[col].value_counts(dropna=False)
        lines.append(f"**{label}**: " + ", ".join(
            f"{'NA' if pd.isna(v) else v}: {c}" for v, c in vc.items()
        ))
    lines.append("")

    # --- Adoption status ---
    lines.append("## Adoption Status (Fig. 1)")
    lines.append("")
    lines.append("| Standard | In use | Preparing | Considering | No plan |")
    lines.append("|----------|--------|-----------|-------------|---------|")
    for std, col in [("ISO 26262", "iso26262_adoption"),
                     ("SOTIF", "sotif_adoption"), ("UL 4600", "ul4600_adoption")]:
        vc = df[col].value_counts()
        lines.append(f"| {std} | {vc.get('In use', 0)} | {vc.get('Preparing', 0)} "
                     f"| {vc.get('Considering', 0)} | {vc.get('No plan', 0)} |")
    lines.append("")

    # ------------------------------------------------------------------
    # RQ1: Perceived benefits and costs
    # ------------------------------------------------------------------
    lines.append("## RQ1: Perceived Benefits and Costs (Table 2)")
    lines.append("")
    lines.append("### Descriptive Statistics")
    lines.append("")
    lines.append("| Item | ISO n | ISO M | ISO SD | SOTIF n | SOTIF M | SOTIF SD |")
    lines.append("|------|-------|-------|--------|---------|---------|----------|")
    for item in BC_COLS:
        ni, mi, si = desc(df[f"iso26262_{item}"])
        ns, ms, ss = desc(df[f"sotif_{item}"])
        lines.append(f"| {BC_LABELS[item]} | {ni} | {ff(mi)} | {ff(si)} "
                     f"| {ns} | {ff(ms)} | {ff(ss)} |")
    lines.append("")

    # Wilcoxon paired ISO vs SOTIF
    lines.append("### Wilcoxon Signed-Rank: ISO 26262 vs SOTIF")
    lines.append("")
    rq1_results = []
    for item in BC_COLS:
        res = wilcoxon_paired(df[f"iso26262_{item}"], df[f"sotif_{item}"])
        rq1_results.append((BC_LABELS[item], res))
    rq1_holm = holm_correction([r["p"] for _, r in rq1_results])

    lines.append("| Item | n(pair) | n(nz) | W | p | p(Holm) | r |")
    lines.append("|------|---------|-------|---|---|---------|---|")
    for i, (label, res) in enumerate(rq1_results):
        lines.append(f"| {label} | {res['n_pair']} | {res['n_nonzero']} "
                     f"| {ff(res['W'],1)} | {fp(res['p'])} "
                     f"| {fp(rq1_holm[i])}{sig(rq1_holm[i])} | {fr(res['r'])} |")
    lines.append("")

    # A-SPICE maturity effect
    lines.append("### A-SPICE Maturity Effect (Mann-Whitney U)")
    lines.append("")
    oem_mask = df["company_type"] == "OEM"
    aspice_op = df["aspice_status"] == "Operating"
    aspice_not = df["aspice_status"].isin(["Not operating", "Preparing"])

    for std_lbl, prefix in [("ISO 26262", "iso26262"), ("SOTIF", "sotif")]:
        lines.append(f"**{std_lbl}**:")
        lines.append("")
        lines.append("| Item | Op. n | Op. M | Not op. n | Not op. M | U | p | p(Holm) | r |")
        lines.append("|------|-------|-------|-----------|-----------|---|---|---------|---|")
        mat_results = []
        for item in BC_COLS:
            col = f"{prefix}_{item}"
            x = df.loc[aspice_op, col]
            y = df.loc[aspice_not, col]
            res = mann_whitney_u(x, y)
            nx, mx, _ = desc(x)
            ny, my, _ = desc(y)
            mat_results.append((BC_LABELS[item], nx, mx, ny, my, res))
        mat_holm = holm_correction([r["p"] for *_, r in mat_results])
        for i, (label, nx, mx, ny, my, res) in enumerate(mat_results):
            lines.append(f"| {label} | {nx} | {ff(mx)} | {ny} | {ff(my)} "
                         f"| {ff(res['U'],1)} | {fp(res['p'])} "
                         f"| {fp(mat_holm[i])}{sig(mat_holm[i])} | {fr(res['r'])} |")
        lines.append("")

    # ------------------------------------------------------------------
    # RQ2: Adoption drivers
    # ------------------------------------------------------------------
    lines.append("## RQ2: Adoption Drivers (Tables 3--4)")
    lines.append("")
    lines.append("### Descriptive Statistics")
    lines.append("")
    lines.append("| Driver | ISO n | ISO M | ISO SD | SOTIF n | SOTIF M | SOTIF SD |")
    lines.append("|--------|-------|-------|--------|---------|---------|----------|")
    for item in DRIVER_COLS:
        ni, mi, si = desc(df[f"iso26262_drv_{item}"])
        ns, ms, ss = desc(df[f"sotif_drv_{item}"])
        lines.append(f"| {DRIVER_LABELS[item]} | {ni} | {ff(mi)} | {ff(si)} "
                     f"| {ns} | {ff(ms)} | {ff(ss)} |")
    lines.append("")

    # Wilcoxon paired (RQ2a)
    lines.append("### RQ2a: Wilcoxon Signed-Rank ISO 26262 vs SOTIF (Table 3)")
    lines.append("")
    rq2a_results = []
    for item in DRIVER_COLS:
        res = wilcoxon_paired(df[f"iso26262_drv_{item}"], df[f"sotif_drv_{item}"])
        rq2a_results.append((DRIVER_LABELS[item], res))
    rq2a_holm = holm_correction([r["p"] for _, r in rq2a_results])

    lines.append("| Driver | n(pair) | n(nz) | W | p | p(Holm) | r |")
    lines.append("|--------|---------|-------|---|---|---------|---|")
    for i, (label, res) in enumerate(rq2a_results):
        lines.append(f"| {label} | {res['n_pair']} | {res['n_nonzero']} "
                     f"| {ff(res['W'],1)} | {fp(res['p'])} "
                     f"| {fp(rq2a_holm[i])}{sig(rq2a_holm[i])} | {fr(res['r'])} |")
    lines.append("")

    # Friedman
    lines.append("### Friedman Test (within-standard driver comparison)")
    lines.append("")
    for std_lbl, prefix in [("ISO 26262", "iso26262_drv"), ("SOTIF", "sotif_drv")]:
        cols_df = df[[f"{prefix}_{d}" for d in DRIVER_COLS]]
        n, chi2, p = friedman_test(cols_df)
        lines.append(f"- **{std_lbl}**: n={n}, chi2={ff(chi2,2)}, p={fp(p)}")
    lines.append("")

    # OEM vs Supplier (RQ2b)
    lines.append("### RQ2b: OEM vs Supplier (Mann-Whitney U, Table 4)")
    lines.append("")
    for std_lbl, prefix in [("ISO 26262", "iso26262_drv"), ("SOTIF", "sotif_drv")]:
        lines.append(f"**{std_lbl}**:")
        lines.append("")
        lines.append("| Driver | OEM n | OEM M | Sup. n | Sup. M | U | p | p(Holm) | r |")
        lines.append("|--------|-------|-------|--------|--------|---|---|---------|---|")
        oem_results = []
        for item in DRIVER_COLS:
            col = f"{prefix}_{item}"
            res = mann_whitney_u(df.loc[oem_mask, col], df.loc[~oem_mask, col])
            no, mo, _ = desc(df.loc[oem_mask, col])
            ns, ms, _ = desc(df.loc[~oem_mask, col])
            oem_results.append((DRIVER_LABELS[item], no, mo, ns, ms, res))
        oem_holm = holm_correction([r["p"] for *_, r in oem_results])
        for i, (label, no, mo, ns, ms, res) in enumerate(oem_results):
            lines.append(f"| {label} | {no} | {ff(mo)} | {ns} | {ff(ms)} "
                         f"| {ff(res['U'],1)} | {fp(res['p'])} "
                         f"| {fp(oem_holm[i])}{sig(oem_holm[i])} | {fr(res['r'])} |")
        lines.append("")

    # ------------------------------------------------------------------
    # RQ3: V-model role allocation
    # ------------------------------------------------------------------
    lines.append("## RQ3: V-Model Role Allocation (Table 5)")
    lines.append("")
    lines.append("### Descriptive Statistics")
    lines.append("")
    lines.append("| Activity | ISO n | ISO M | ISO SD | SOTIF n | SOTIF M | SOTIF SD |")
    lines.append("|----------|-------|-------|--------|---------|---------|----------|")
    for item in ROLE_COLS:
        ni, mi, si = desc(df[f"iso26262_role_{item}"])
        ns, ms, ss = desc(df[f"sotif_role_{item}"])
        lines.append(f"| {ROLE_LABELS[item]} | {ni} | {ff(mi)} | {ff(si)} "
                     f"| {ns} | {ff(ms)} | {ff(ss)} |")
    lines.append("")

    # Wilcoxon paired (RQ3a) + boundary clarity (RQ3b) — single Holm family (6 tests)
    lines.append("### RQ3a: Wilcoxon Signed-Rank ISO 26262 vs SOTIF")
    lines.append("")
    rq3a_results = []
    for item in ROLE_COLS:
        res = wilcoxon_paired(df[f"iso26262_role_{item}"], df[f"sotif_role_{item}"])
        rq3a_results.append((ROLE_LABELS[item], res))

    # RQ3b: boundary clarity
    boundary = df["sotif_boundary_clarity"]
    b_res = one_sample_wilcoxon(boundary, median=3.0)

    # Holm correction over all 6 RQ3 tests (5 role + 1 boundary)
    rq3_pvals = [r["p"] for _, r in rq3a_results] + [b_res["p"]]
    rq3_holm = holm_correction(rq3_pvals)

    lines.append("| Activity | n(pair) | n(nz) | W | p | p(Holm) | r |")
    lines.append("|----------|---------|-------|---|---|---------|---|")
    for i, (label, res) in enumerate(rq3a_results):
        lines.append(f"| {label} | {res['n_pair']} | {res['n_nonzero']} "
                     f"| {ff(res['W'],1)} | {fp(res['p'])} "
                     f"| {fp(rq3_holm[i])}{sig(rq3_holm[i])} | {fr(res['r'])} |")
    lines.append("")

    lines.append("### RQ3b: SOTIF Boundary Clarity (One-Sample Wilcoxon)")
    lines.append("")
    nb, mb, sb = desc(boundary)
    ambig = int(((boundary == 1) | (boundary == 2)).sum())
    rq3b_holm_p = rq3_holm[5]  # 6th test in the family
    lines.append(f"- n = {nb}, mean = {ff(mb)}, SD = {ff(sb)}")
    lines.append(f"- Rated 'less clear' or 'much less clear': {ambig} ({ambig*100//nb}%)")
    lines.append(f"- One-sample Wilcoxon against median=3: "
                 f"W={ff(b_res['W'],1)}, n_nonzero={b_res['n_nonzero']}, "
                 f"p={fp(b_res['p'])}, p(Holm)={fp(rq3b_holm_p)}{sig(rq3b_holm_p)}, "
                 f"r={fr(b_res['r'])}")
    lines.append("")

    # ------------------------------------------------------------------
    # Multi-standard challenges
    # ------------------------------------------------------------------
    lines.append("## Multi-Standard Challenges (Table 6)")
    lines.append("")
    challenge_counts = {}
    for val in df["multi_standard_challenges"].dropna():
        for part in str(val).split("; "):
            part = part.strip()
            if part:
                challenge_counts[part] = challenge_counts.get(part, 0) + 1
    sorted_ch = sorted(challenge_counts.items(), key=lambda x: -x[1])
    lines.append("| Challenge | Count |")
    lines.append("|-----------|-------|")
    for ch, cnt in sorted_ch:
        lines.append(f"| {ch} | {cnt} |")
    lines.append("")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    lines.append("## Summary: Holm-Corrected Significant Findings")
    lines.append("")
    lines.append("The following four findings survive Holm correction (p < 0.05):")
    lines.append("")

    cr = rq2a_results[3][1]
    lines.append(f"1. **Contract requirements** (RQ2a): ISO > SOTIF, "
                 f"W={ff(cr['W'],1)}, p={fp(cr['p'])}, "
                 f"p(Holm)={fp(rq2a_holm[3])}, r={fr(cr['r'])}")

    ic = rq2a_results[4][1]
    lines.append(f"2. **Industry conformity** (RQ2a): ISO > SOTIF, "
                 f"W={ff(ic['W'],1)}, p={fp(ic['p'])}, "
                 f"p(Holm)={fp(rq2a_holm[4])}")

    # Recalculate ISO contract OEM/Supplier for summary
    cr_mwu = mann_whitney_u(df.loc[oem_mask, "iso26262_drv_contract_requirements"],
                            df.loc[~oem_mask, "iso26262_drv_contract_requirements"])
    iso_oem_pvals = []
    for item in DRIVER_COLS:
        col = f"iso26262_drv_{item}"
        iso_oem_pvals.append(mann_whitney_u(
            df.loc[oem_mask, col], df.loc[~oem_mask, col])["p"])
    iso_oem_holm = holm_correction(iso_oem_pvals)
    lines.append(f"3. **ISO 26262 contract requirements: OEM vs Supplier** (RQ2b): "
                 f"Supplier > OEM, U={ff(cr_mwu['U'],1)}, p={fp(cr_mwu['p'])}, "
                 f"p(Holm)={fp(iso_oem_holm[3])}, r={fr(cr_mwu['r'])}")

    lines.append(f"4. **SOTIF boundary clarity** (RQ3b): below neutral, "
                 f"W={ff(b_res['W'],1)}, p={fp(b_res['p'])}, "
                 f"p(Holm)={fp(rq3b_holm_p)}, r={fr(b_res['r'])}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce statistical analyses from survey data"
    )
    parser.add_argument("--input", default="data/responses.csv",
                        help="Path to responses CSV (default: data/responses.csv)")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(path)
    # Convert Likert/role/boundary "NA" strings to NaN
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace("NA", np.nan)
        # Convert numeric-like columns
        if col.endswith(("_safety", "_accountability", "_requirements",
                         "_conformity", "_improvement", "_burden",
                         "_effectiveness")) or "_role_" in col or \
           col.endswith("_boundary_clarity"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(run_analysis(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
