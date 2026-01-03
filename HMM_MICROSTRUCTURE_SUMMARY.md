# Microstructure-Enhanced HMM: Research Summary

**Notebook:** [14_hmm_microstructure_regimes.ipynb](notebooks/14_hmm_microstructure_regimes.ipynb)
**Date:** January 2, 2026
**Sample Period:** 2015-2024
**Observations:** 2,488 trading days

---

## Model Specification

### Enhanced HMM Structure

**States:** 3 (Orderly/Informational, Liquidity-driven/Tradable, Disorderly/Crisis)
**Covariance:** Diagonal (parsimonious)
**Estimation:** Baum-Welch EM with K-means initialization

### Input Feature Vector

$$X_t = \begin{bmatrix} \text{VIX}_t \\ |\\Delta S_t| \\ \text{RealizedVol}_t^{(10d)} \\ \Delta^2 S_t \\ \log(\text{Volume}_{\text{LQD},t}) \end{bmatrix}$$

**Features:**

1. **VIX:** Equity market stress (macro indicator)
2. **$|\\Delta S_t|$:** Absolute daily spread change (speed of movement)
3. **RealizedVol$_t^{(10d)}$:** 10-day rolling std of spread changes (microstructure volatility)
4. **$\\Delta^2 S_t$:** Spread acceleration (flow pressure proxy)
5. **$\\log(\text{Volume}_{\text{LQD},t})$:** LQD ETF trading volume (liquidity proxy)

All features standardized (z-scored).

### Signal (Not an HMM Input)

**Credit spread signal:** BAMLC0A0CM (ICE BofA US Corporate Index OAS)

Used exclusively for mean reversion testing, NOT included in HMM feature set.

---

## Regime Identification Results

### Regime Distribution

| Regime | Days | % | VIX Mean | Spread Mean | |ΔS| Mean | Interpretation |
|--------|------|---|----------|-------------|----------|----------------|
| **0** | 1,124 | 45.2% | 13.5 | 1.20 bps | 0.0056 bps | Orderly/Informational |
| **1** | 1,175 | 47.2% | 20.4 | 1.27 bps | 0.0109 bps | Liquidity-driven/Tradable |
| **2** | 189 | 7.6% | 32.9 | 1.79 bps | 0.0492 bps | Disorderly/Crisis |

**Key characteristics:**

**State 0 (Orderly/Informational):**
- Lowest VIX (13.5), minimal spread volatility (0.76 bps/10d)
- Low absolute spread changes (0.56 bps/day on average)
- Normal LQD volume (log-volume = 15.65)
- Low LQD absolute returns (0.25% daily)
- **45% of sample period**

**State 1 (Liquidity-driven/Tradable):**
- Moderate VIX (20.4), elevated spread volatility (1.35 bps/10d)
- Moderate spread changes (1.09 bps/day)
- Elevated LQD volume (log-volume = 16.34)
- Moderate LQD absolute returns (0.37% daily)
- **47% of sample period (largest regime)**

**State 2 (Disorderly/Crisis):**
- High VIX (32.9), extreme spread volatility (5.39 bps/10d)
- Large spread changes (4.92 bps/day)
- Very high LQD volume (log-volume = 16.77)
- Large LQD absolute returns (0.80% daily)
- **8% of sample period (rare events)**

---

## Mean Reversion Validation

### 10-Day Horizon Results

| Regime | Beta | SE | t-stat | p-value | Half-Life | N |
|--------|------|----|----|---------|-----------|---|
| Unconditional | -0.0685 | 0.0091 | -7.54 | < 0.001 | 9.8 days | 2,478 |
| **State 0** | **-0.000003** | 0.0085 | **-0.0004** | **0.9997** | **>200k days** | 1,114 |
| **State 1** | **-0.0526** | 0.0122 | **-4.32** | **< 0.001** | **12.8 days** | 1,165 |
| **State 2** | **-0.3873** | 0.0308 | **-12.56** | **< 0.001** | **1.4 days** | 179 |

### Key Findings

**1. State 0 shows NO mean reversion:**
- Beta ≈ 0 (statistically indistinguishable from zero)
- p-value = 0.9997 (cannot reject random walk)
- Half-life > 200,000 days (essentially infinite)
- **45% of trading days have NO statistical edge for mean reversion**

**2. State 1 exhibits moderate mean reversion:**
- Beta = -0.053 (significant at p < 0.001)
- Half-life = 12.8 days (reasonable trading horizon)
- **This is the "tradable" regime**
- 47% of trading days

**3. State 2 shows very strong mean reversion:**
- Beta = -0.387 (highly significant, t = -12.56)
- Half-life = 1.4 days (rapid convergence)
- Consistent with temporary liquidity dislocations
- But only 8% of trading days

---

## Comparison: Macro-Only vs Microstructure-Enhanced

### Regime Distribution

|  | Macro-Only | Microstructure | Change |
|---|------------|----------------|--------|
| **State 0 (Low Vol)** | 47.1% | 45.2% | -1.9pp |
| **State 1 (Moderate)** | 43.4% | 47.2% | +3.8pp |
| **State 2 (High Vol)** | 9.5% | 7.6% | -1.9pp |

The microstructure model identifies **slightly more State 1 days** (tradable liquidity-driven regime).

### Half-Life Estimates (10-Day Horizon)

| Regime | Macro-Only | Microstructure | Interpretation |
|--------|------------|----------------|----------------|
| **State 0** | No MR (581k days) | No MR (216k days) | Both show NO mean reversion |
| **State 1** | 13.9 days | 12.8 days | Similar moderate MR |
| **State 2** | 1.5 days | 1.4 days | Similar strong MR |

**Results are highly consistent between models.** The choice between macro-only and microstructure-enhanced depends on:
- Data availability (microstructure requires LQD volume/returns)
- Regime interpretability (microstructure directly measures flow pressure)
- Real-time implementation (microstructure variables may have slight lags)

---

## Economic Interpretation

### Why State 1 is "Tradable"

State 1 (Liquidity-driven/Tradable) exhibits:
- **Elevated but not extreme flow:** Moderate LQD volume and spread changes
- **Consistent mean reversion:** 12.8-day half-life is actionable
- **Nearly half the sample:** 47% of days provide trading opportunities
- **VIX 15-25 range:** Not calm (State 0) nor crisis (State 2)

This regime likely captures:
- Normal market stress episodes (Fed announcements, earnings seasons)
- Temporary credit spread widening due to ETF flows
- Post-stress normalization periods
- Tactical rebalancing opportunities

### Why State 0 Has No Mean Reversion

State 0 characteristics suggest efficient price discovery:
- Low volatility → small deviations from equilibrium
- Normal flow → no forced buying/selling
- VIX < 15 → complacent markets
- **Spreads follow a random walk** in stable environments

### Why State 2 Mean Reversion is Deceptive

State 2 shows the strongest mean reversion (1.4-day half-life) but:
- **Execution costs spike:** Bid-ask spreads widen materially during stress
- **Liquidity disappears:** Large trades move the market
- **Tail risk:** Extreme events can persist longer than expected
- **Only 8% of days:** Rare regime, hard to systematically exploit

---

## Transition Dynamics

The microstructure-enhanced model exhibits similar transition properties to the macro-only version:

**Expected regime duration:**
- State 0: ~75 days (persistent orderly periods)
- State 1: ~45 days (moderate duration)
- State 2: ~40 days (crisis episodes last ~2 months)

**Sequential transitions:** Regimes transition State 0 ↔ State 1 ↔ State 2 (no direct jumps)

**Asymmetric stress exits:** State 2 exits to State 1 (not directly to State 0)

---

## Key Conclusions

### 1. Microstructure Variables Add Value

The enhanced model successfully incorporates:
- Spread dynamics (speed, acceleration, volatility)
- Market flow pressure (LQD volume, returns)
- Direct liquidity signals

These variables improve regime interpretability and align with market microstructure theory.

### 2. Consistent with Macro-Only Model

Both approaches identify:
- **State 0:** No mean reversion (45-47% of days)
- **State 1:** Moderate mean reversion (43-47% of days)
- **State 2:** Strong mean reversion (8-10% of days)

The microstructure model provides **similar statistical results** with **enhanced economic interpretation**.

### 3. State 1 is the Target Regime for Trading

If attempting to trade mean reversion:
- **Focus on State 1** (liquidity-driven/tradable)
- **Avoid State 0** (no statistical edge)
- **Be cautious in State 2** (execution costs may exceed statistical profits)

### 4. Unconditional Mean Reversion Tests are Misleading

Pooling all regimes yields:
- Unconditional half-life: 9.8 days
- Unconditional beta: -0.069

But this masks:
- 45% of days with NO mean reversion (State 0)
- 47% with moderate mean reversion (State 1)
- 8% with strong mean reversion (State 2)

**Trading unconditionally means trading 45% of the time with no edge.**

---

## Implementation Recommendations

### For Trading Strategy Development

**If building a mean reversion strategy:**

1. **Use microstructure-enhanced HMM** if LQD data is available
   - Direct flow pressure signals
   - Clearer identification of liquidity-driven regimes
   - Real-time interpretability

2. **Use macro-only HMM** if microstructure data is unavailable
   - Produces very similar regime structure
   - Slightly simpler implementation
   - Fewer data dependencies

3. **Trade ONLY in State 1** (liquidity-driven regime)
   - 47% of days provide opportunity
   - 12.8-day half-life is actionable
   - Flow pressure creates exploitable dislocations

4. **Avoid trading in State 0** (orderly regime)
   - No statistical edge (β ≈ 0, p > 0.99)
   - 45% of days
   - Spreads follow random walk

5. **Exercise extreme caution in State 2** (crisis regime)
   - Strong mean reversion statistically
   - But execution costs, liquidity risk, tail risk are severe
   - Only 8% of days

### Critical Question Remains

**Even in State 1, can mean reversion profits exceed transaction costs?**

- State 1 mean reversion: ~5 bps over 12.8 days
- Round-trip transaction costs: ~5 bps (LQD 3 bps + IEF 2 bps)
- **Profit margin is razor-thin before costs**

This question is deferred to the trading proof-of-concept phase.

---

## Files Generated

### Notebooks
- [14_hmm_microstructure_regimes.ipynb](notebooks/14_hmm_microstructure_regimes.ipynb)

### Results
- `results/hmm_microstructure/regime_characteristics.csv`
- `results/hmm_microstructure/transition_matrix.csv`
- `results/hmm_microstructure/mean_reversion_tests.csv`
- `results/hmm_microstructure/half_life_estimates.csv`

### Figures
- `results/figures/hmm_microstructure_timeseries.png`
- `results/figures/hmm_microstructure_mean_reversion.png`

---

## Final Recommendation

**The microstructure-enhanced HMM successfully identifies regime-dependent mean reversion dynamics.**

**Use this model to:**
1. Identify State 1 (liquidity-driven/tradable regime) for potential trading
2. Avoid State 0 (no mean reversion) to conserve capital
3. Monitor State 2 (crisis) for extreme events but avoid active trading

**Next step:** Map BAMLC0A0CM signal to LQD/IEF execution in a trading proof-of-concept, focusing on State 1 regime.

**Critical test:** Can State 1 mean reversion profits (5-6 bps over 13 days) exceed transaction costs (5 bps round-trip)?

If not, the statistical edge is real but not economically exploitable.

---

**End of Summary**
