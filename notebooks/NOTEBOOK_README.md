# Quant Research Notebook: Mean Reversion in IG Credit Spreads

## Summary

This notebook presents a clean, professional analysis of mean reversion in investment-grade credit spreads, conditioned on regime identification via Gaussian Hidden Markov Model (HMM).

## Structure

The notebook is organized into **11 cells** (3 markdown, 8 code):

1. **Title + Research Question** (Markdown)
   - Clearly states the hypothesis: mean reversion is regime-dependent
   - Methods overview and data sources

2. **Setup & Data Loading** (Code)
   - Imports, configuration, paths
   - Loads HMM-processed data and regime summary
   - Prints diagnostic info on regimes and sample size

3. **Regime Characteristics** (Markdown)
   - Transition matrix and persistence

4. **Regime Summary Table** (Code)
   - Displays regime distribution and persistence
   - Shows High Stress regime is rare (3%) but critical

5. **Time Series Visualization** (Code)
   - Credit spread over time
   - VIX with regime shading (2008 GFC, 2020 COVID highlighted)
   - Saved to `results/tables/regime_timeseries.png`

6. **Methodology** (Markdown)
   - Explains median regression (robust to outliers)
   - Bootstrap procedure for confidence intervals

7. **Primary Test: Normal Regime, 10-Day Horizon** (Code)
   - Bootstrap median regression with B=1,000
   - Outputs:
     - Median β = -0.165, 95% CI [-0.200, -0.125]
     - Median half-life ≈ 4 days
   - Saves results to `results/tables/mean_reversion_primary.json`

8. **Bootstrap Visualization** (Code)
   - Histogram of β distribution
   - Boxplot of half-life distribution
   - Saved to `results/tables/bootstrap_primary_result.png`

9. **Cross-Regime Comparison** (Markdown)
   - Extends to all three regimes and multiple horizons

10. **Multi-Regime Results** (Code)
    - Compares β coefficients across horizons (5d, 10d, 21d)
    - Shows mean reversion weakens in High Stress regime

11. **Conclusions** (Markdown)
    - Key findings
    - Refined hypothesis
    - Limitations and next steps
    - **Ends with actionable recommendations**

## Key Outputs

### Numeric Results
- **Normal regime, 10-day horizon:**
  - Median β = **-0.1651** (95% CI: [-0.2000, -0.1250])
  - Median half-life = **4.2 days** (95% CI: [3.5, 5.5])
  - Interpretation: **Strong, significant mean reversion**

### Visualizations
- `regime_timeseries.png` — Credit spread and VIX with regime shading
- `bootstrap_primary_result.png` — β distribution and half-life boxplot

### JSON Summary
- `mean_reversion_primary.json` — Machine-readable results for downstream use

## Design Principles

1. **Presentation-Ready**: Minimal clutter, no exploratory dead-ends, no comments on code quality
2. **Robustness**: Uses median regression (resistant to outliers) and bootstrap inference
3. **Conservative**: Conclusions limited to what data support; acknowledges High Stress sample size limitation
4. **Narrative Flow**: Each cell logically follows the previous; reads like a research paper
5. **Reproducible**: All paths relative; all code is self-contained and documented

## How to Use

### To view the notebook:
```bash
cd /Users/felixguex/credit-spread-tracer
jupyter notebook notebooks/03_mean_reversion_analysis.ipynb
```

### To export to HTML/PDF:
```bash
jupyter nbconvert --to html notebooks/03_mean_reversion_analysis.ipynb --output mean_reversion_report.html
```

### To extract results table for a paper:
```bash
cat results/tables/mean_reversion_primary.json | python -m json.tool
```

## For PM / Internal Review

- **Key result**: Normal regime (36% of sample) shows significant mean reversion at 10-day horizon with tight confidence intervals. The finding is robust (median regression) and well-powered (n≈401).
- **Regime differences**: Mean reversion weakens in crises (High Stress); current sample insufficient to quantify precisely (n≈37).
- **Trade-off**: The notebook sacrifices exhaustive robustness checks (Cook's distance, Huber regression, outlier diagnostics) for clarity and brevity; these can be added if needed for publication.
- **Next priority**: Validate High Stress regime with dealer balance-sheet data (SLOOS, leverage); test out-of-sample on 2021–2024.

---

**Created**: December 2025  
**Data**: 2007–2025 (1,103 daily observations)  
**Regimes**: 3 (Gaussian HMM via EM)  
**Method**: Median regression + bootstrap (B=1,000)
