# Mean Reversion in Investment-Grade Credit Spreads

**Research Question**: Is mean reversion in investment-grade credit spreads conditional on dealer balance-sheet constraints?

## �� Quick Start

### View the Analysis

```bash
jupyter notebook notebooks/03_mean_reversion_analysis.ipynb
```

This notebook contains:
- HMM regime identification (3 regimes)
- Mean-reversion tests by regime (median regression)
- Bootstrap confidence intervals (B=1,000)
- Cross-regime robustness checks

**Runtime**: ~2–3 minutes (bootstrap included)

## 📊 Key Results

**Normal Regime** (36% of sample, n=401):
- **β = -0.165** (95% CI: [-0.200, -0.125]) ✅ Significantly negative
- **Half-life ≈ 4 days** (95% CI: [3.5, 5.5]) — fast mean reversion
- **Conclusion**: Strong, reliable mean reversion

**Low Stress Regime** (60%, n=665):
- Weaker mean reversion (smaller |β|)

**High Stress Regime** (3%, n=37, crises only):
- Insufficient data (mean reversion weakens or breaks down)

## 📁 Project Structure

```
notebooks/
  └─ 03_mean_reversion_analysis.ipynb     ← Main analysis (START HERE)
data/processed/
  └─ full_processed_data_hmm.csv          (HMM regimes + features)
results/tables/
  ├─ regime_summary.csv
  ├─ regime_transitions.csv
  ├─ conditional_tests.csv
  ├─ bootstrap_normal_10d_summary.json
  └─ [plots & additional outputs]
HANDOFF.md                                 ← Full project summary
```

## 🔍 Methodology

**Data**: BAMLC0A0CM (IG spread), VIX, STLFSI, realized vol; 1,103 obs (2007–2025)

**Regimes**: Gaussian HMM (3 regimes identified via EM algorithm)
- Low Stress (60%), Normal (36%), High Stress (3%)
- Features: VIX, STLFSI, realized vol, order flow, interactions

**Regression**: Median regression (quantile q=0.5) + Bootstrap (B=1,000)
- Robust to outliers (fat-tailed financial data)
- Nonparametric 95% confidence intervals

**Model**: ΔS_{t+h} = α + β S_t + ε
- H₀: β = 0 vs H₁: β < 0 (mean reversion)

## 📋 Setup

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/03_mean_reversion_analysis.ipynb
```

## 📚 Documentation

- **`HANDOFF.md`** — Executive summary & project overview
- **`notebooks/INDEX.md`** — Notebook guide & navigation
- **`notebooks/NOTEBOOK_README.md`** — Cell-by-cell breakdown

## ⏭️ Next Steps

1. **Validate regime labels** — Confirm High Stress = 2008 GFC, 2020 COVID
2. **Out-of-sample test** — Forecast 2021–2024 data; measure realized returns
3. **Model extensions** — Add controls; test 4-regime HMM; cross-market validation

---

**Status**: ✅ Ready for presentation | **Last Updated**: December 2025
