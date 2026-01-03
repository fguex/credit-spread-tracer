# LQD-IEF Credit Spread Mean Reversion Strategy
## Final Analysis Summary (2015-2025)

**Author:** Claude Sonnet 4.5
**Date:** January 2, 2026
**Project:** credit-spread-tracer

---

## Executive Summary

This project tested a **mean reversion trading strategy** on the LQD-IEF credit spread using HMM regime detection and Ornstein-Uhlenbeck process modeling. After extensive testing across multiple periods (2015-2025) with various signal generation methods, the **conclusion is unambiguous: this strategy does not work in practice.**

**Bottom Line:** Even with **zero transaction costs**, the best strategy made only **$36 over 5 years** on $1M capital. With realistic costs (3-5 bps), the strategy loses money in all periods tested.

---

## 1. Hypothesis & Statistical Tests

### Original Hypothesis
Credit spreads (LQD - IEF) exhibit mean-reverting behavior that can be exploited for trading, particularly when conditioned on market stress regimes detected via Hidden Markov Models.

### Statistical Validation Results

**Unconditional Mean Reversion Tests (2015-2025):**
- ✅ **Horizon 10d:** Sharpe -2.26, p=0.024 (significant **)
- ✅ **Horizon 21d:** Sharpe -4.55, p<0.001 (highly significant ***)
- **Conclusion:** Mean reversion exists statistically at longer horizons

**Regime-Conditional Tests:**
- **Low Stress (Regime 0):**
  - 5d: Sharpe -2.31, p=0.021 (**)
  - 10d: Sharpe -2.88, p=0.004 (***)
  - 21d: Sharpe -3.97, p<0.001 (***)

- **High Stress (Regime 1):**
  - 10d: Sharpe -4.05, p<0.001 (***)
  - 21d: Sharpe -8.58, p<0.001 (***)

**Key Finding:** Statistical mean reversion is **stronger during high stress regimes**, but this does NOT translate into tradeable alpha.

---

## 2. Spread Characteristics By Period

| Period | Mean (bps) | Std (bps) | Min (bps) | Max (bps) | Range (bps) |
|--------|------------|-----------|-----------|-----------|-------------|
| **2015-2019** | 1.33 | 0.24 | 0.90 | 2.21 | 1.31 |
| **2020-2024** | 1.23 | 0.38 | 0.77 | 4.01 | 3.24 |
| **2025** | 0.86 | 0.10 | 0.74 | 1.21 | 0.47 |

**Key Observations:**
1. **Spread compression:** Mean fell from 1.33 bps (2015-19) to 0.86 bps (2025) - a 35% decline
2. **Volatility collapse:** Std fell from 0.24 bps to 0.10 bps in 2025 - a 58% decline
3. **2020-2024 anomaly:** COVID spike created temporary volatility, but mean reverted quickly

---

## 3. Backtest Results Summary

### Methodology
- **Zero transaction costs** (to test pure signal quality)
- **Capital:** $1,000,000
- **Position size:** 50% of capital per leg
- **Regime filtering:** No entry in high stress, forced exit on regime change
- **VIX filter:** No entry if VIX > 30

### Best Strategy: Absolute Levels
- **Logic:** Short spread when > 1.5 bps, long when < 0.9 bps, exit at 1.2 bps
- **Reason:** Simple, interpretable, best risk-adjusted returns

### Cross-Period Performance (Absolute Levels Strategy, Zero Costs)

| Period | Return | Sharpe | Max DD | Trades | Win Rate | P&L |
|--------|--------|--------|--------|--------|----------|-----|
| **2015-2019** | 0.0036% | **1.48** | -0.002% | 6 | 60.2% | **+$36** |
| **2020-2024** | 0.0013% | **0.62** | -0.0005% | 4 | 52.5% | **+$13** |
| **2025** | -0.0001% | **-0.12** | -0.0007% | 1 | 47.9% | **-$1** |
````markdown
# Final summary — credit-spread-tracer (minimal)

Bottom line: statistical mean reversion exists in the LQD–IEF spread but is not economically tradeable. With realistic transaction costs and current spread magnitudes the tested strategies produce negligible or negative P&L.

What we found
- Statistical mean reversion is present (e.g., normal regime, 10-day horizon: β ≈ -0.165, half-life ≈ 4 days).
- Mean reversion is stronger in stressed periods but those periods are rare and the current backtest filters often avoid them.
- The best practical signal (simple absolute-level thresholds) produced trivial P&L (tens of dollars on $1M) under zero-cost assumptions.

Key limitations
- Spread magnitude and volatility are too small vs realistic transaction costs (3–5 bps round-trip) to generate tradeable profits.
- OU parameter estimation is unstable across long rolling windows; short windows improve stability but not profitability.
- Regime-sample size for high-stress periods is small, limiting confidence in regime-conditional performance.

Recommendation
- Do not deploy this strategy. Archive exploratory notebooks and keep the presentation-ready analysis and validation artifacts for reproducibility.

Files kept
- `notebooks/02_hmm_validation_results.ipynb`
- `notebooks/03_mean_reversion_analysis.ipynb`
- `notebooks/INDEX.md` and `notebooks/NOTEBOOK_README.md` (navigation and docs)

If you want me to permanently delete the archived notebooks (remove from git history), tell me and I will proceed. Otherwise the deletions below remove them from the working tree but they remain recoverable in git history.

````
- **Total:** 10 bps per round trip
