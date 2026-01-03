# HMM Regime Detection Research: Executive Summary

**Document:** [12_hmm_regime_validation.ipynb](notebooks/12_hmm_regime_validation.ipynb)
**Date:** January 2, 2026
**Analysis Period:** 2015-2024

---

## Objective

Establish statistical foundation for regime-conditional credit spread mean reversion using Hidden Markov Model applied to macro-financial conditions.

---

## Key Findings

### 1. Regime Identification

A 2-state Gaussian HMM successfully identifies distinct market regimes:

**Low Stress Regime (Regime 0):**
- 85.4% of observations (2,218 days)
- Lower VIX, tighter spreads, stable financial conditions
- Slower mean reversion dynamics

**High Stress Regime (Regime 1):**
- 14.6% of observations (377 days)
- Elevated VIX, wider spreads, deteriorating financial conditions
- **Dramatically stronger mean reversion**

### 2. Mean Reversion Strength by Regime

**10-Day Forecast Horizon (Median Regression with Bootstrap):**

| Regime | Beta Coefficient | t-statistic | p-value | Interpretation |
|--------|------------------|-------------|---------|----------------|
| Unconditional | -0.0588 | -7.24 | < 0.001 | Moderate mean reversion |
| **Low Stress** | **-0.0286** | -3.48 | < 0.001 | Weak mean reversion |
| **High Stress** | **-0.3526** | -12.56 | < 0.001 | **Very strong mean reversion** |

**Critical Finding:** Mean reversion is **12.3x stronger** in the high stress regime.

### 3. Statistical Robustness

- **Method:** Median regression (robust to outliers and fat tails)
- **Inference:** Bootstrap confidence intervals (distribution-free, 1000 replications)
- **Consistency:** Results hold across all forecast horizons (5-63 days)
- **Significance:** All tests reject random walk at p < 0.001

### 4. Half-Life Estimates

**10-Day Horizon:**
- Low Stress: 241 days (very slow mean reversion)
- High Stress: 16 days (rapid mean reversion)

The 15x difference in half-lives confirms regime-dependent dynamics.

---

## Economic Interpretation

### Why Mean Reversion Strengthens During Stress

The finding that mean reversion **accelerates** during high stress is consistent with market microstructure theory:

1. **Liquidity dislocation:** Forced selling creates temporary mispricings
2. **Fundamental reassertion:** Credit fundamentals reassert as liquidity normalizes
3. **Larger deviations:** Wider spreads provide greater absolute convergence potential
4. **Price discovery:** Extreme moves trigger informed trading and reversion

**Critically:** This does NOT imply stress periods are optimal for trading. Execution costs, liquidity impact, and risk management are separate considerations.

---

## Signal Specification

Based on this analysis, the mean reversion signal is defined as:

**Signal:** BAMLC0A0CM (BofA Merrill Lynch US Corporate Master OAS)

**Regime Conditioning:** HMM state $R_t \in \{0, 1\}$ modulates expected dynamics but does NOT directly enter signal construction.

**Why BAMLC0A0CM:**
- Market-wide credit spread index
- Economically interpretable
- Not contaminated by ETF-specific mechanics
- Clean separation between signal and execution

---

## Conclusions

### Established Results

1. **Credit spread mean reversion exists** (unconditional p < 0.001)
2. **Mean reversion is state-dependent** (12x stronger in high stress)
3. **HMM successfully identifies regimes** with distinct dynamics
4. **Statistical methodology is robust** (median regression + bootstrap)
5. **Stress ≠ no mean reversion** (common misconception rejected)

### What This Does NOT Establish

This research notebook does **NOT** address:
- Trading execution mechanics
- Transaction costs
- Position sizing
- Risk management
- Actual profitability

These are explicitly deferred to the trading proof-of-concept phase.

---

## Next Steps: Trading Implementation

The transition to trading requires:

### Signal-to-Execution Mapping
- BAMLC0A0CM is an index (not directly tradable)
- Execute via LQD (corporate bond ETF) + IEF/TLT (rates hedge)
- Estimate hedge ratios from returns regression

### Trading Rules
- Entry/exit thresholds (z-score or OU-based)
- Position sizing (constant vs volatility-scaled)
- Stop-loss and holding period limits

### Cost Modeling
- LQD bid-ask: ~3 bps
- IEF bid-ask: ~2 bps
- Rebalancing frequency and costs

### Risk Monitoring
- Maximum drawdown limits
- Regime-conditional exposure controls
- Daily rehedging for duration neutrality

---

## Files Generated

### Notebooks
- [12_hmm_regime_validation.ipynb](notebooks/12_hmm_regime_validation.ipynb) - Main research analysis

### Results
- `results/hmm_research/regime_mean_reversion_tests.csv` - Statistical test results
- `results/figures/hmm_regime_timeseries.png` - Regime identification over time
- `results/figures/hmm_mean_reversion_by_regime.png` - Beta coefficients by regime

---

## Recommendation

**This research validates the statistical foundation for regime-conditional mean reversion.**

Proceed to trading proof-of-concept with:
- Signal: BAMLC0A0CM
- Regime filter: 2-state HMM on macro features
- Execution: LQD/IEF with duration-neutral hedging
- Focus: Cost modeling and realistic P&L attribution

The statistical justification is now established and publication-ready.

---

**End of Summary**
