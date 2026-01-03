# Three-State Hidden Markov Model: Research Summary

**Notebook:** [13_hmm_three_state_validation.ipynb](notebooks/13_hmm_three_state_validation.ipynb)
**Date:** January 2, 2026
**Sample Period:** 2015-2024
**Observations:** 2,604 trading days

---

## Model Specification

### HMM Structure

**States:** 3 (Low Volatility, Intermediate, High Volatility)
**Features:** 4 macro-financial indicators
**Covariance:** Diagonal (parsimonious assumption)
**Estimation:** Baum-Welch EM algorithm with K-means initialization

### Input Features

1. **Credit spread volatility:** 20-day rolling std of BAMLC0A0CM changes
2. **Equity market stress:** VIX index
3. **Rates volatility:** 20-day rolling std of 10Y yield changes (proxy for MOVE)
4. **Financial conditions:** ANFCI (Adjusted National Financial Conditions Index)

All features standardized (z-scored) prior to estimation.

### Signal (Not an HMM Input)

**Credit spread signal:** BAMLC0A0CM (ICE BofA US Corporate Index OAS)

Used exclusively for mean reversion testing, not included in HMM feature set.

---

## Regime Identification Results

### Regime Characteristics

| Regime | Days | Pct | VIX Mean | VIX Std | Spread Vol | Spread Mean | Spread Std |
|--------|------|-----|----------|---------|------------|-------------|------------|
| **State 0 (Low Vol)** | 1,226 | 47.1% | 15.4 | 4.4 | 0.0085 bps | 1.13 bps | 0.18 bps |
| **State 1 (Intermediate)** | 1,130 | 43.4% | 18.5 | 5.3 | 0.0136 bps | 1.33 bps | 0.27 bps |
| **State 2 (High Vol)** | 248 | 9.5% | 31.2 | 11.2 | 0.0504 bps | 1.75 bps | 0.52 bps |

**Interpretation:**

**State 0:** Stable market conditions. Low VIX (15.4), tight credit spreads (1.13 bps), minimal volatility. Represents normal, low-stress environments.

**State 1:** Moderate stress. Elevated VIX (18.5), wider spreads (1.33 bps), increased volatility. Transition regime between stability and stress.

**State 2:** Crisis-like conditions. High VIX (31.2), wide spreads (1.75 bps), extreme volatility. Represents acute stress episodes (10% of sample).

### Transition Dynamics

**Transition Probability Matrix:**

|  | To State 0 | To State 1 | To State 2 |
|---|------------|------------|------------|
| **From State 0** | 98.8% | 1.2% | ~0% |
| **From State 1** | 1.4% | 98.1% | 0.5% |
| **From State 2** | ~0% | 2.0% | 98.0% |

**Key findings:**

1. **High persistence:** All states exhibit strong persistence (98%+ probability of remaining in same state)

2. **Sequential transitions:** Regimes transition sequentially (0 ↔ 1 ↔ 2), with virtually no direct jumps between State 0 and State 2

3. **Expected duration:**
   - State 0: 81 days (persistent low volatility)
   - State 1: 53 days (moderate duration)
   - State 2: 49 days (stress episodes last ~2 months)

4. **Asymmetric exits:** State 2 exits almost exclusively to State 1 (not directly to State 0), suggesting gradual stress normalization rather than sudden reversals

---

## Mean Reversion Validation

### Methodology

For each regime and horizon $h$, we estimate:

$$\Delta S_{t,t+h} = \alpha + \beta S_t + \epsilon_t$$

using median regression (robust to outliers) with bootstrap confidence intervals (1,000 replications).

### 10-Day Horizon Results

| Regime | Beta | SE | t-stat | p-value | Half-Life | N |
|--------|------|----|----|---------|-----------|---|
| **Unconditional** | -0.0588 | 0.0082 | -7.15 | < 0.001 | 11.4 days | 2,594 |
| **State 0 (Low Vol)** | -0.000001 | 0.0064 | -0.0002 | 0.9999 | **581k days** | 1,216 |
| **State 1 (Intermediate)** | -0.0488 | 0.0116 | -4.22 | < 0.001 | 13.9 days | 1,120 |
| **State 2 (High Vol)** | -0.3709 | 0.0290 | -12.80 | < 0.001 | **1.5 days** | 238 |

### Key Findings

**1. State-dependent mean reversion:**

Mean reversion strength varies dramatically across regimes:
- **State 0:** Essentially no mean reversion (β ≈ 0, statistically insignificant)
- **State 1:** Moderate mean reversion (β = -0.049, p < 0.001)
- **State 2:** Very strong mean reversion (β = -0.371, p < 0.001)

**2. Half-life differentials:**

The ratio of half-lives is extreme:
- State 2 vs State 0: **388,000x faster** mean reversion
- State 2 vs State 1: **9.3x faster** mean reversion

**3. Statistical significance:**

- Unconditional and State 1/2 tests reject random walk at p < 0.001
- State 0 shows no evidence of mean reversion (p = 0.9999)
- Bootstrap confidence intervals confirm robustness

**4. Economic interpretation:**

State 2 (high volatility) exhibits rapid mean reversion (1.5-day half-life), consistent with temporary liquidity dislocations that quickly normalize. State 0 (low volatility) shows no mean reversion, suggesting spreads follow a random walk in stable environments.

---

## Conclusions

### Established Results

**1. Parsimonious regime classification:**

The three-state HMM successfully identifies distinct macro-financial environments using only four observable features.

**2. Regime-dependent mean reversion:**

Credit spread mean reversion is not a constant property. It is:
- **Absent** in low volatility regimes (State 0)
- **Moderate** in intermediate regimes (State 1)
- **Very strong** in high volatility regimes (State 2)

**3. Statistical robustness:**

Results are robust across:
- Multiple forecast horizons (5-63 days)
- Distribution-free bootstrap inference
- Median regression (outlier-robust)

**4. Transition structure:**

Regimes exhibit:
- High persistence (98%+ auto-transition probability)
- Sequential dynamics (0 ↔ 1 ↔ 2)
- Asymmetric stress exits (State 2 → State 1 → State 0)

### Critical Interpretation

**State 0 finding is crucial:** The absence of mean reversion in low volatility regimes suggests that:
- Trading mean reversion during calm periods is statistically unjustified
- Mean reversion strategies should be regime-conditional
- Unconditional mean reversion tests (pooling all regimes) can be misleading

**State 2 finding has dual implications:**
- Strong mean reversion creates statistical opportunity
- But high volatility implies elevated execution costs, wider spreads, and potential liquidity risk
- Statistical mean reversion ≠ tradeable opportunity

### What This Model Is NOT

This HMM is not:
- A timing model
- A trading signal
- A predictor of spread direction
- Optimized for Sharpe ratio

It is a **conditioning layer** that identifies when mean reversion dynamics are materially different.

---

## Implications for Trading

### Signal Specification

**Credit spread signal:** BAMLC0A0CM (BofA US Corporate OAS Index)

**Regime filter:** 3-state HMM based on macro-financial stress indicators

### Regime-Conditional Trading Rules

**State 0 (Low Vol):**
- Do not trade mean reversion (no statistical edge)
- Consider momentum or carry strategies instead
- 47% of sample period

**State 1 (Intermediate):**
- Moderate mean reversion (13.9-day half-life)
- Conservative position sizing
- 43% of sample period

**State 2 (High Vol):**
- Strong mean reversion (1.5-day half-life)
- Potential opportunity, but high execution risk
- Only 10% of sample period

### Open Questions for Trading Implementation

**1. Execution mapping:**
- How to map BAMLC0A0CM to LQD/IEF positions?
- Hedge ratio estimation (duration neutrality)
- Basis risk between index and ETF

**2. Cost modeling:**
- Are bid-ask spreads (3-5 bps) smaller than mean reversion profits?
- Rebalancing costs for daily duration hedging
- Slippage during State 2 (high vol, low liquidity)

**3. Risk management:**
- Position sizing conditional on regime
- Drawdown limits in State 2
- Regime transition risk (false signals)

These questions are explicitly deferred to the trading proof-of-concept phase.

---

## Files Generated

### Notebooks
- [13_hmm_three_state_validation.ipynb](notebooks/13_hmm_three_state_validation.ipynb)

### Results
- `results/hmm_three_state/regime_characteristics.csv` - Regime summary statistics
- `results/hmm_three_state/transition_matrix.csv` - State transition probabilities
- `results/hmm_three_state/mean_reversion_tests.csv` - Full regression results
- `results/hmm_three_state/half_life_estimates.csv` - Half-life by regime

### Figures
- `results/figures/hmm_three_state_timeseries.png` - Regime identification over time
- `results/figures/hmm_three_state_mean_reversion.png` - Beta coefficients by regime

---

## Recommendation

**The three-state HMM provides a clean statistical foundation for regime-conditional mean reversion analysis.**

Key takeaway: **Mean reversion only exists in stressed environments (States 1 and 2).** Trading strategies should explicitly condition on regime, as pooled/unconditional approaches mix distinct dynamics.

Proceed to trading proof-of-concept with:
- Signal: BAMLC0A0CM
- Regime filter: 3-state HMM (focus on States 1 and 2)
- Execution: LQD/IEF with duration-neutral hedging
- Critical test: Can mean reversion profits exceed transaction costs in States 1/2?

---

**End of Summary**
