# Technical Note: Feature Standardization in Regression Analysis

## Why Standardize?

When running OLS regression with multiple predictors on different scales, standardization is critical:

### Problem with Unstandardized Features

Consider our microstructure model:
```
ΔS_{t+1} = α + β₀·z_t + β₁·z_t×stress_t + β₂·momentum_t + β₃·vol_t + ε
```

**Original scales**:
- `z_score`: Already standardized (mean ≈ 0, std ≈ 1)
- `z_stress_interaction`: Product of standardized variables (mean ≈ 0, std varies)
- `momentum`: Raw basis points (std ≈ 5-20 bps)
- `realized_vol`: Raw basis points (std ≈ 2-10 bps)

### Issues Without Standardization

1. **Non-comparable coefficients**: 
   - β₂ (momentum) in units of "Δbps per bps of momentum"
   - β₃ (volatility) in units of "Δbps per bps of volatility"
   - Cannot directly compare magnitudes

2. **Numerical instability**:
   - Features on vastly different scales can cause convergence issues
   - Condition number of X'X matrix increases

3. **Misleading inference**:
   - Standard errors affected by feature scale
   - Hard to assess relative importance

### Solution: Standardize All Features

Transform each feature X to:
```
X_std = (X - mean(X)) / std(X)
```

**After standardization**:
- All features have mean = 0, std = 1
- Coefficients represent: "Change in Y per 1-SD change in X"
- Direct comparison of coefficient magnitudes

### Example

**Before standardization**:
- β₂ = -0.05: "Each 1 bps increase in momentum → 0.05 bps decrease in spread change"
- β₃ = -2.00: "Each 1 bps increase in volatility → 2 bps decrease in spread change"
- **Cannot compare**: Is momentum or volatility more important?

**After standardization**:
- β₂ = -0.30: "Each 1-SD increase in momentum → 0.30 bps decrease in spread change"
- β₃ = -0.50: "Each 1-SD increase in volatility → 0.50 bps decrease in spread change"
- **Can compare**: Volatility has stronger effect (|β₃| > |β₂|)

## Implementation

### In Microstructure Analysis Notebook

```python
from sklearn.preprocessing import StandardScaler

# Prepare features
X_cols = ['z_score', 'z_stress_interaction', 'momentum', 'realized_vol']
reg_df_clean = reg_df[['delta_S'] + X_cols].dropna()

# Standardize
scaler = StandardScaler()
X_standardized = pd.DataFrame(
    scaler.fit_transform(reg_df_clean[X_cols]),
    index=reg_df_clean.index,
    columns=X_cols
)

# Run regression with standardized features
results = run_conditional_regression(
    y=reg_df_clean['delta_S'],
    X=X_standardized,
    robust_se=True
)
```

### Verification

After standardization:
```python
print(X_standardized.mean())  # All ≈ 0
print(X_standardized.std())   # All ≈ 1
```

## Impact on Results

### What Changes:
- **Coefficient values**: Scaled by feature std deviation
- **Coefficient magnitudes become comparable**
- **Coefficient standard errors**: Scaled similarly

### What Doesn't Change:
- **t-statistics**: Remain the same
- **p-values**: Remain the same
- **R²**: Remains the same
- **Statistical significance**: Unchanged
- **Model fit**: Identical

## When to Standardize

### Always standardize when:
1. Multiple predictors on different scales
2. Comparing coefficient magnitudes across features
3. Using regularization (Ridge, Lasso, Elastic Net)
4. Interpreting feature importance

### Optional for:
1. Single predictor models (scaling doesn't affect inference)
2. When coefficient interpretation in original units is critical
3. Binary predictors (though standardization still helps comparison)

## Best Practices

1. **Fit scaler on training data only** (if doing train/test split)
2. **Transform both train and test** using same scaler
3. **Document standardization** in analysis
4. **Report coefficients in standardized units** with clear labels
5. **Provide interpretation** of 1-SD changes in original units if needed

## References

- Greene, W. (2018). *Econometric Analysis* (8th ed.). Section 4.4.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Section 3.4.
- Gelman, A., & Hill, J. (2006). *Data Analysis Using Regression*. Chapter 4.

## Summary

**Standardization is essential for multivariate regression** when:
- Features have different scales
- Coefficient comparison is needed
- Numerical stability matters

Our microstructure analysis implements standardization for all features before OLS, ensuring:
- Interpretable, comparable coefficients
- Robust numerical computation
- Clear identification of most important predictors
