# Companion Repository: Adoption of Safety Standards in the Japanese Automotive Industry

This repository provides reproducibility materials for the paper:

> Y. Matsuno, S. Ochiai, F. Kono, "Adoption of Safety Standards in the Japanese Automotive Industry," submitted to SafeComp 2026.

## Contents

| Path | Description |
|------|-------------|
| `questionnaire.md` | Full survey questionnaire (English/Japanese bilingual) |
| `data/responses.csv` | Individual-level survey responses (n=30, fully anonymized, English) |
| `scripts/analyze.py` | Reproduces all statistical tests reported in the paper |
| `scripts/generate_figures.py` | Generates Fig. 1 (adoption status) |

## Data

The survey was conducted anonymously within the JASPAR functional safety working group. `data/responses.csv` contains all 30 individual responses with timestamps and identifying metadata removed. All Japanese responses (including free-text comments) have been translated to English.

### responses.csv

Each row is one respondent (n=30). Columns include:

- **Profile**: `company_type`, `role`, `aspice_status`, `iso26262_experience`
- **Adoption status**: `{standard}_adoption` (In use / Preparing / Considering / No plan)
- **Adoption drivers** (Likert 1-5): `{standard}_drv_{driver}`
- **Benefits/costs** (Likert 1-5): `{standard}_{item}`
- **V-model roles** (1=OEM only ... 5=Supplier only): `{standard}_role_{activity}`
- **Boundary clarity** (1=much less clear ... 5=much more clear): `sotif_boundary_clarity`
- **Free-text comments**: `{standard}_comments`, `assurance_case_comments`, etc.
- **Multi-standard**: `multi_standard_challenges`, `multi_standard_policy`

Missing or not-applicable values are empty (NaN).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Reproduce all statistical analyses
python scripts/analyze.py

# Generate Fig. 1
python scripts/generate_figures.py
```

### Requirements

- Python 3.9+
- numpy >= 1.21
- scipy >= 1.7
- pandas >= 1.3
- matplotlib >= 3.4

## Mapping: Survey Items to Paper Tables/Figures

| Paper element | Survey items | CSV columns |
|---------------|-------------|-------------|
| Table 1 (Profile) | Q1--Q4 | `company_type`, `role`, `aspice_status`, `iso26262_experience` |
| Fig. 1 (Adoption) | Q5, Q22, Q38 | `*_adoption` |
| Table 2 (RQ1: Benefits/Costs) | Q12--Q14, Q28--Q30 | `*_safety_improvement`, `*_operational_burden`, `*_cost_effectiveness` |
| Table 2 (RQ1: Maturity effect) | Q12--Q14 x Q3 | benefit/cost columns grouped by `aspice_status` |
| Table 3 (RQ2a: Drivers) | Q7--Q11, Q23--Q27 | `*_drv_*` columns |
| Table 4 (RQ2b: OEM/Supplier) | Q7--Q11 x Q1, Q23--Q27 x Q1 | driver columns grouped by `company_type` |
| Table 5 (RQ3a: V-model roles) | Q15--Q19, Q31--Q35 | `*_role_*` columns |
| Section 5 (RQ3b: Boundary) | Q36 | `sotif_boundary_clarity` |
| Table 6 (Challenges) | Q57 | `multi_standard_challenges` |

## Statistical Methods

All tests follow the methodology described in Section 4 of the paper:

- **Wilcoxon signed-rank test** (paired comparisons: ISO vs SOTIF)
  - Uses non-zero differences only (`zero_method="wilcox"`)
  - Exact p-values for n_nonzero <= 25
- **Mann-Whitney U test** (independent groups: OEM vs Supplier, A-SPICE maturity)
  - Tie-corrected variance for effect size calculation
- **One-sample Wilcoxon** (boundary clarity against neutral median = 3)
- **Holm correction** applied within each test family
- **Effect size** r = |Z| / sqrt(N); reported only when n_nonzero >= 10

## License

The data and scripts in this repository are provided for academic reproducibility purposes.
