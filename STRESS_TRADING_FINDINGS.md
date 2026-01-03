# Stress Trading with Risk-Adjusted Position Sizing

**Date:** January 2, 2026
**Notebook:** [11_stress_trading_risk_adjusted.ipynb](notebooks/11_stress_trading_risk_adjusted.ipynb)

---

## Executive Summary

**Answer to "Should we trade during stress?":** It depends on the period.

- **2015-2019:** Trading during stress with **25% position size** slightly improves Sharpe (1.43 vs 1.37)
- **2020-2024:** Avoiding stress entirely performs better (Sharpe 0.34 vs 0.29-0.31)
- **Net impact:** Marginal - gains are tiny even with zero costs

---

## Test Configuration

Previously, the strategy **completely blocked trading during high stress**:
- No entries when regime = 1 (high stress)
- Forced exit of existing positions when regime changed to 1

Now tested **4 approaches**:

1. **No Stress (Original):** Avoid stress entirely (baseline)
2. **Stress 50% Size:** Trade during stress with full 50% position
3. **Stress 25% Size:** Trade during stress with half-size 25% position (risk-adjusted)
4. **Stress 10% Size:** Trade during stress with minimal 10% position (very conservative)

All tests use the **inverted OU strategy** (window=126, entry=2.5, exit=0.7) which performed best in previous analysis.

---

## Results Summary (Zero Transaction Costs)

### 2015-2019 Period

| Configuration | Sharpe | P&L | Trades | % Time in Stress | Max DD |
|---------------|--------|-----|--------|------------------|--------|
| No Stress (Original) | 1.37 | $35.50 | 375 | 0.0% | -0.50 bps |
| Stress 50% Size | 1.37 | $42.00 | 439 | 12.9% | -1.15 bps |
| **Stress 25% Size** | **1.43** | **$39.00** | 439 | 12.9% | **-0.72 bps** |
| Stress 10% Size | 1.42 | $37.20 | 439 | 12.9% | -0.50 bps |

**Winner:** Risk-adjusted 25% sizing
- **+4.4% better Sharpe** than avoiding stress
- **+$3.50 more P&L** (9.9% gain)
- **Lower drawdown** than 50% sizing
- **64 more trades** (captures more opportunities)

---

### 2020-2024 Period

| Configuration | Sharpe | P&L | Trades | % Time in Stress | Max DD |
|---------------|--------|-----|--------|------------------|--------|
| **No Stress (Original)** | **0.34** | **$7.00** | 205 | 0.0% | **-0.80 bps** |
| Stress 50% Size | 0.29 | $11.50 | 330 | 42.9% | -2.15 bps |
| Stress 25% Size | 0.31 | $8.25 | 330 | 42.9% | -1.07 bps |
| Stress 10% Size | 0.30 | $6.30 | 330 | 42.9% | -0.75 bps |

**Winner:** No stress trading (original approach)
- **Better Sharpe** than stress trading (0.34 vs 0.29-0.31)
- **Lower drawdown** than stress approaches
- **Fewer trades** (205 vs 330) - less overtrading
- Higher P&L with stress trading, but **worse risk-adjusted returns**

**Key Observation:** 2020-2024 had **43% of trading days in high stress** (COVID, rate hikes), making stress avoidance more beneficial.

---

### 2025 Period

All configurations performed identically:
- Sharpe: -0.87
- P&L: -$3
- Trades: 32

**Reason:** 2025 had minimal stress events, so stress vs no-stress made no difference.

---

## Combined Analysis (2015-2024)

### Total P&L (10 Years, Zero Costs)

| Configuration | 2015-2019 | 2020-2024 | **Total** | Trades |
|---------------|-----------|-----------|-----------|--------|
| No Stress | $35.50 | $7.00 | **$42.50** | 580 |
| Stress 50% | $42.00 | $11.50 | **$53.50** | 769 |
| **Stress 25%** | $39.00 | $8.25 | **$47.25** | 769 |
| Stress 10% | $37.20 | $6.30 | **$43.50** | 769 |

**Observation:** Stress trading increases total P&L by **$3.75-11** over 10 years (7-26% gain).

### Risk-Adjusted Returns

| Configuration | Avg Sharpe | Trades | Trades/Year |
|---------------|------------|--------|-------------|
| **No Stress** | **0.85** | 580 | 58 |
| Stress 50% | 0.83 | 769 | 77 |
| Stress 25% | **0.87** | 769 | 77 |
| Stress 10% | 0.86 | 769 | 77 |

**Winner by Sharpe:** Risk-adjusted 25% approach (0.87 avg)

---

## Key Findings

### 1. Period-Dependent Performance

**2015-2019 (Low stress environment):**
- Stress trading with 25% sizing **improves** Sharpe by 4.4%
- Only 12.9% of trading time in stress
- Captures opportunities without excessive risk

**2020-2024 (High stress environment):**
- Avoiding stress **outperforms** (Sharpe 0.34 vs 0.29-0.31)
- 42.9% of trading time would be in stress
- Stress avoidance prevents bad trades during volatility

### 2. Risk Adjustment Works

The **25% stress position sizing**:
- Reduces max drawdown vs 50% sizing
- Improves Sharpe vs both 50% and 10% sizing in 2015-2019
- Provides good balance between opportunity and risk

### 3. Still Not Tradeable

Even the best configuration made only **$47.25 over 10 years** on $1M capital.

**With realistic transaction costs (3-5 bps):**
- 769 trades × $2,500/trade = **-$1.9M in costs**
- Net P&L: **-$1.85M** ❌

The strategy **loses heavily** even with improved stress handling.

---

## Stress Exposure Analysis

### Time in Market During High Stress

| Configuration | 2015-2019 | 2020-2024 | Average |
|---------------|-----------|-----------|---------|
| No Stress | 0% | 0% | 0% |
| All Stress Configs | 12.9% | 42.9% | 27.9% |

**Interpretation:**
- 2015-2019 had relatively calm markets (only 12.9% stress exposure possible)
- 2020-2024 had turbulent markets (42.9% stress exposure possible)
- Trading during stress significantly increases exposure in volatile periods

### Trade Count Impact

| Period | No Stress | With Stress | Increase |
|--------|-----------|-------------|----------|
| 2015-2019 | 375 | 439 | **+64 (+17%)** |
| 2020-2024 | 205 | 330 | **+125 (+61%)** |

**Observation:** Stress trading dramatically increases trade frequency, especially in volatile periods. This **amplifies transaction costs**.

---

## Conclusions

### 1. Should You Trade During Stress?

**Short answer:** Marginal benefit, period-dependent.

**Long answer:**
- In calm markets (2015-2019): Small improvement with risk adjustment
- In volatile markets (2020-2024): Better to avoid stress entirely
- Overall: Adds ~$5-10 over 10 years on $1M (negligible)

### 2. What's the Optimal Stress Position Size?

**If trading during stress:** Use **25% position size** (half normal)

**Why:**
- Best Sharpe ratio (0.87 vs 0.83-0.86)
- Lower drawdown than 50% sizing
- Better risk management than full-size positions
- Captures opportunities without excessive risk

### 3. Does This Fix the Strategy?

**No.** Even with optimal stress handling:
- Total P&L: $47.25 over 10 years on $1M (0.0047% gain)
- Transaction costs: -$1.9M
- Net result: **-$1.85M loss**

The fundamental problem remains: **spread movements are too small relative to transaction costs**.

---

## Recommendations

### For This Strategy: Don't Trade It

Even with:
- ✅ Inverted signals (momentum works better than mean reversion)
- ✅ Risk-adjusted stress sizing
- ✅ Optimal hyperparameters from grid search

The strategy **still loses money** after transaction costs.

### If You Must Trade It

**Best configuration (theoretical only):**
- Method: Inverted OU process
- Window: 126 days
- Entry: z > 2.5, Exit: z < 0.7
- Position sizing:
  - Normal periods: 50%
  - High stress: **25%** (risk-adjusted)
- Expected result: Lose ~$1.85M after costs ❌

### Better Alternatives

1. **Wait for extreme stress** (VIX > 40, spread > 3 bps) and trade large moves only
2. **Use options** on credit ETFs instead of spot (lower transaction costs)
3. **Trade less liquid credit** where spreads are wider
4. **Abandon mean reversion entirely** - momentum/trend following might work better

---

## Technical Notes

### Implementation Details

**Risk-adjusted position sizing logic:**
```python
if regime == 1:  # High stress
    current_notional = capital * stress_position_size  # 25%
else:
    current_notional = capital * position_size  # 50%
```

**Position notional updates:**
- Recalculated on every trade entry
- Uses regime at entry time
- P&L calculated using notional from previous day

### Files Created

- [notebooks/11_stress_trading_risk_adjusted.ipynb](notebooks/11_stress_trading_risk_adjusted.ipynb)
- [results/stress_trading/risk_adjusted_comparison.csv](results/stress_trading/risk_adjusted_comparison.csv)
- [results/figures/stress_trading_risk_adjusted.png](results/figures/stress_trading_risk_adjusted.png)

---

## Final Verdict

**Question:** Should we trade during stress with risk-adjusted position sizing?

**Answer:** In theory, yes marginally. In practice, no because the strategy doesn't work anyway.

**Evidence:**
- Risk-adjusted stress trading adds ~$5-10 over 10 years (before costs)
- After transaction costs: loses ~$1.85M
- The stress handling improvement is **completely overwhelmed** by trading costs

**Recommendation:** Stop trying to optimize this strategy. The fundamental economics don't work.

---

**End of Analysis**
