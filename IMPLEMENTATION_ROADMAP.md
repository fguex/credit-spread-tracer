# Implementation Roadmap: Prediction & Trading Strategy

## Phase 1: Prediction Module (`src/prediction/`)

### File 1: `regime_detector.py`
**Purpose**: Daily cluster assignment (lookahead-safe)

```python
class RegimeDetector:
    def __init__(self, n_clusters=4):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    def fit(self, df_train, features):
        """Fit scaler + KMeans on training window only"""
        X_train = df_train[features].dropna()
        self.scaler.fit(X_train)
        self.kmeans.fit(self.scaler.transform(X_train))
    
    def predict(self, df_test, features):
        """Assign cluster to out-of-sample data"""
        X_test = df_test[features].dropna()
        X_scaled = self.scaler.transform(X_test)
        return self.kmeans.predict(X_scaled)
```

### File 2: `model.py`
**Purpose**: Ridge regression with regime-aware features

```python
class RegimeAwarePrediction:
    def __init__(self, horizon=5):
        self.horizon = horizon
        self.model = Ridge(alpha=1.0)
        self.detector = RegimeDetector(n_clusters=4)
    
    def fit(self, df_train, target_col='y_{t+1}'):
        """Train on historical data (train window)"""
        # Assign clusters to training data
        clusters = self.detector.predict(df_train)
        
        # Create cluster dummies
        for c in range(4):
            df_train[f'cluster_{c}'] = (clusters == c).astype(int)
        
        # Ridge with cluster features
        feat_cols = base_features + [f'cluster_{c}' for c in range(4)]
        X = df_train[feat_cols].dropna()
        y = df_train.loc[X.index, target_col]
        
        self.model.fit(X, y)
    
    def predict(self, df_test):
        """Generate daily predictions (expanding window)"""
        predictions = []
        
        for date in df_test.index:
            # Cluster assignment
            cluster = self.detector.predict_single(date)
            
            # Ridge prediction
            pred = self.model.predict([df_test.loc[date, feat_cols]])
            
            # Regime-aware inversion (conditional)
            if cluster in [0, 1, 2]:  # Mean-reversion clusters
                pred = -pred
            
            predictions.append({
                'date': date,
                'cluster': cluster,
                'prediction': pred,
                'inverted': cluster in [0, 1, 2]
            })
        
        return pd.DataFrame(predictions)
```

---

## Phase 2: Trading Strategy Module (`src/trading/`)

### File 1: `backtest.py`
**Purpose**: Expanding-window backtest with transaction costs

```python
def expanding_window_backtest(df, model, train_end='2018-12-31', costs_bp=1.0):
    """
    For each date after train_end:
    1. Train model on all data up to date
    2. Generate prediction
    3. Compute PnL (position = -sign(pred) × actual)
    4. Deduct transaction costs
    """
    
    results = []
    
    for date in df[df.index > pd.Timestamp(train_end)].index:
        # Expanding window
        df_train = df[df.index <= date]
        model.fit(df_train)
        
        # Predict
        pred = model.predict(df.loc[date])
        
        # Position & PnL
        position = -np.sign(pred)
        actual = df.loc[date, 'y_{t+1}']
        gross_pnl = position * actual
        
        # Transaction cost
        cost = costs_bp / 10000  # bp to decimal
        net_pnl = gross_pnl - cost
        
        results.append({
            'date': date,
            'cluster': pred['cluster'],
            'prediction': pred['prediction'],
            'position': position,
            'actual': actual,
            'gross_pnl': gross_pnl,
            'cost': cost,
            'net_pnl': net_pnl
        })
    
    return pd.DataFrame(results)
```

### File 2: `analysis.py`
**Purpose**: Strategy performance metrics

```python
def compute_metrics(backtest_results):
    """PnL, hit ratio, Sharpe, max drawdown by cluster"""
    
    metrics = {}
    
    for cluster in [0, 1, 2, 3]:
        mask = backtest_results['cluster'] == cluster
        results_c = backtest_results[mask]
        
        # Profitability
        cum_pnl = results_c['net_pnl'].cumsum()
        metrics[f'cluster_{cluster}'] = {
            'n_trades': len(results_c),
            'cum_pnl': cum_pnl.iloc[-1],
            'mean_pnl': results_c['net_pnl'].mean(),
            'std_pnl': results_c['net_pnl'].std(),
            'sharpe': results_c['net_pnl'].mean() / results_c['net_pnl'].std() * np.sqrt(252),
            'max_dd': (cum_pnl - cum_pnl.cummax()).min(),
            'hit_ratio': (np.sign(results_c['prediction']) == np.sign(results_c['actual'])).mean()
        }
    
    return metrics
```

---

## Phase 3: Configuration & Orchestration

### File: `config.py`

```python
# Model hyperparameters
RIDGE_ALPHA = 1.0
N_CLUSTERS = 4
HORIZON = 5  # days ahead

# Features
BASE_FEATURES = ['S_t', 'ΔS_t', 'DGS10_t', 'SP500_ret_20d', 'VIX_t']
CLUSTERING_FEATURES = ['S_t', 'ΔS_t', 'VIX_t', 'YC_slope_t', 'vol_dS_10d']

# Backtesting
TRAIN_START = '2016-01-01'
TRAIN_END = '2018-12-31'
TEST_END = '2026-12-31'

# Costs (basis points per round-trip)
COSTS_BP = {
    'retail': 1.0,
    'institutional': 0.5,
    'prime_brokerage': 0.25
}

# Regimes
MEAN_REVERSION_CLUSTERS = [0, 1, 2]
MOMENTUM_CLUSTERS = [3]
```

### File: `main.py`

```python
def run_full_pipeline(horizon=5, cost_scenario='institutional'):
    """End-to-end backtest"""
    
    # 1. Load data
    df = build_dataset(start='2016-01-01', mode='daily', h=horizon)
    
    # 2. Initialize model
    model = RegimeAwarePrediction(horizon=horizon)
    model.detector.fit(
        df[df.index <= '2018-12-31'],
        features=CLUSTERING_FEATURES
    )
    
    # 3. Backtest
    cost_bp = COSTS_BP[cost_scenario]
    results = expanding_window_backtest(df, model, costs_bp=cost_bp)
    
    # 4. Analyze
    metrics = compute_metrics(results)
    
    # 5. Report
    print(f"Horizon: h={horizon}")
    print(f"Cost Scenario: {cost_scenario} ({cost_bp} bp)")
    print("\nPer-Cluster Performance:")
    for cluster, m in metrics.items():
        print(f"  {cluster}: {m['cum_pnl']:.2f} PnL, {m['sharpe']:.2f} Sharpe")
    
    return results, metrics
```

---

## Testing Checklist

- [ ] Regime detector produces stable cluster assignments
- [ ] Ridge model trains on expanding windows without errors
- [ ] Predictions include regime-aware inversion logic
- [ ] PnL calculation matches expected position * actual formula
- [ ] Hit ratio computation excludes zero actuals (strict masking)
- [ ] Transaction costs properly deducted from gross PnL
- [ ] Backtest generates results for all horizons (h=1,5,10,15,20)
- [ ] Per-cluster analysis shows regime-aware advantage
- [ ] Sharpe ratio improves with regime conditioning (target: +0.5 vs baseline)

---

## Expected Output Files

```
results/
├── predictions_h5_retail.csv          (daily predictions, 1 bp costs)
├── predictions_h5_institutional.csv   (daily predictions, 0.5 bp costs)
├── backtest_summary_h5.json           (PnL, metrics by cluster)
├── trading_performance_h5.csv         (trade-level breakdown)
└── regime_analysis.json               (statistical tests, regime stats)
```

---

## Success Criteria

1. **Regime identification**: 4 clusters with clear separation (silhouette score > 0.4)
2. **Predictive power**: Hit ratio > 45% in at least 2 clusters
3. **Profitability**: Net PnL > 0 for h=5 at institutional cost level (0.5 bp)
4. **Risk management**: Sharpe ratio > 0.5, max drawdown < 20%
5. **Reproducibility**: All results regenerable from code (no hard-coded values)

---

## Timeline Estimate

| Task | Est. Hours |
|------|-----------|
| Regime Detector | 2-3 |
| Ridge Model + Inversion Logic | 3-4 |
| Expanding-Window Backtest | 2-3 |
| Performance Analysis & Metrics | 2-3 |
| Testing & Validation | 3-4 |
| Documentation & Cleanup | 2 |
| **Total** | **14-20 hours** |

---

**Status**: Ready for implementation. Regime analysis is complete and findings are robust. Proceeding to production code.
