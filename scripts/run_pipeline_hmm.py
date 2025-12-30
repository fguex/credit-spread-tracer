"""
Production pipeline with HMM-based regime identification.

Automated workflow:
1. Fetch data from FRED
2. Clean and align data
3. Engineer HMM features (realized vol, stress indicators)
4. Estimate regimes via Gaussian HMM
5. Test mean-reversion by regime
6. Save results

Run: python scripts/run_pipeline_hmm.py [--fetch]
"""

import argparse
from pathlib import Path
import pandas as pd
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import START_DATE, RESULTS_TABLES
from src.data.fetch_fred import fetch_all_series, save_raw_data, load_raw_data
from src.data.clean import clean_data, compute_summary_stats
from src.data.hmm_features import prepare_hmm_features, save_hmm_features
from src.models.hmm_regimes import estimate_regimes_hmm, compute_regime_statistics
from src.models.regimes import (
    summarize_regimes,
    compute_regime_transition_matrix,
    validate_regime_counts
)
from src.models.regressions import (
    run_regime_conditional_tests,
    run_unconditional_test,
    interpret_results
)

# Results directory
RESULTS_DIR = Path(RESULTS_TABLES)


def main(fetch_fresh_data: bool = False):
    """
    Run complete HMM-based analysis pipeline.
    
    Parameters
    ----------
    fetch_fresh_data : bool
        If True, download latest data from FRED
    """
    print("=" * 80)
    print("CREDIT SPREAD MEAN-REVERSION ANALYSIS (HMM-BASED REGIMES)")
    print("=" * 80)
    
    # =========================================================================
    # Stage 1: Data Acquisition
    # =========================================================================
    print("\n[1/6] Data Acquisition")
    print("-" * 40)
    
    if fetch_fresh_data:
        print("Fetching data from FRED...")
        series_dict = fetch_all_series(start=START_DATE)
        save_raw_data(series_dict)
    else:
        print("Using cached data...")
    
    raw_data = load_raw_data()
    print(f"Loaded {len(raw_data)} observations, {len(raw_data.columns)} series")
    print(f"Date range: {raw_data.index.min()} to {raw_data.index.max()}")
    
    # =========================================================================
    # Stage 2: Data Cleaning
    # =========================================================================
    print("\n[2/6] Data Cleaning")
    print("-" * 40)
    
    clean = clean_data(raw_data)
    print(f"After cleaning: {len(clean)} observations")
    
    # Save summary statistics
    summary_stats = compute_summary_stats(clean)
    summary_stats.to_csv(RESULTS_DIR / "data_summary.csv")
    print("Saved summary statistics")
    
    # =========================================================================
    # Stage 3: HMM Feature Engineering
    # =========================================================================
    print("\n[3/6] HMM Feature Engineering")
    print("-" * 40)
    
    # Prepare features for HMM
    features, scaler = prepare_hmm_features(clean)
    print(f"Prepared {features.shape[1]} features for HMM")
    print(f"Features: {features.columns.tolist()}")
    print(f"Valid observations: {len(features)}")
    
    # Save features
    save_hmm_features(features)
    
    # =========================================================================
    # Stage 4: HMM Regime Estimation
    # =========================================================================
    print("\n[4/6] HMM Regime Estimation")
    print("-" * 40)
    
    # Estimate regimes
    feature_cols = features.columns.tolist()
    regimes, hmm_model, hmm_scaler = estimate_regimes_hmm(
        features,
        feature_cols,
        stress_col="vix",  # Sort regimes by VIX
        verbose=True
    )
    
    print(f"\nEstimated regimes for {len(regimes)} periods")
    print(f"Regime distribution:")
    print(regimes.value_counts().sort_index())
    
    # Validate regime counts
    validate_regime_counts(regimes)
    
    # Compute regime statistics
    regime_stats = compute_regime_statistics(regimes, features)
    regime_stats.to_csv(RESULTS_DIR / "hmm_regime_statistics.csv", index=False)
    print("\nSaved HMM regime statistics")
    
    # Regime summary and transitions
    regime_summary = summarize_regimes(regimes)
    regime_transitions = compute_regime_transition_matrix(regimes)
    
    regime_summary.to_csv(RESULTS_DIR / "regime_summary.csv", index=False)
    regime_transitions.to_csv(RESULTS_DIR / "regime_transitions.csv")
    
    print(f"\nRegime Transition Matrix:")
    print(regime_transitions)
    
    # Save HMM model parameters
    hmm_params = pd.DataFrame({
        "n_regimes": [hmm_model.n_components],
        "n_features": [hmm_model.n_features],
        "converged": [hmm_model.monitor_.converged],
        "n_iter": [hmm_model.monitor_.iter],
        "log_likelihood": [hmm_model.score(features.values)]
    })
    hmm_params.to_csv(RESULTS_DIR / "hmm_model_info.csv", index=False)
    
    # Save transition matrix separately
    transmat = pd.DataFrame(
        hmm_model.transmat_,
        columns=[f"To Regime {i}" for i in range(hmm_model.n_components)],
        index=[f"From Regime {i}" for i in range(hmm_model.n_components)]
    )
    transmat.to_csv(RESULTS_DIR / "hmm_transition_matrix.csv")
    
    # =========================================================================
    # Stage 5: Mean-Reversion Tests
    # =========================================================================
    print("\n[5/6] Mean-Reversion Tests")
    print("-" * 40)
    
    # Align spread with regimes
    spread = clean.loc[regimes.index, "spread"]
    
    # Unconditional tests
    print("\nRunning unconditional tests...")
    from src.config import HORIZONS
    unconditional_results = run_unconditional_test(spread, HORIZONS)
    unconditional_results = interpret_results(unconditional_results)
    unconditional_results.to_csv(RESULTS_DIR / "unconditional_tests.csv", index=False)
    print(unconditional_results[['horizon', 'beta', 'p_beta', 'mean_reverting']])
    
    # Conditional tests by regime
    print("\nRunning regime-conditional tests...")
    conditional_results = run_regime_conditional_tests(spread, regimes, HORIZONS)
    conditional_results = interpret_results(conditional_results)
    conditional_results.to_csv(RESULTS_DIR / "conditional_tests.csv", index=False)
    
    # Summary by regime
    for regime in sorted(regimes.unique()):
        regime_res = conditional_results[conditional_results['regime'] == regime]
        sig_horizons = regime_res[regime_res['mean_reverting']]['horizon'].tolist()
        print(f"  Regime {regime}: mean reversion at horizons {sig_horizons}")
    
    # =========================================================================
    # Stage 6: Save Processed Data
    # =========================================================================
    print("\n[6/6] Saving Processed Data")
    print("-" * 40)
    
    # Combine everything
    processed = clean.loc[regimes.index].copy()
    processed["regime"] = regimes
    
    # Add HMM features
    for col in features.columns:
        if col not in processed.columns:
            processed[f"hmm_{col}"] = features[col]
    
    processed.to_csv("data/processed/full_processed_data_hmm.csv")
    print(f"Saved processed data: data/processed/full_processed_data_hmm.csv")
    print(f"Shape: {processed.shape}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE (HMM-BASED)")
    print("=" * 80)
    
    print("\nKey Findings:")
    print("-" * 40)
    
    regime_names = {0: "Low Stress (High Liquidity)", 1: "Normal", 2: "High Stress (Low Liquidity)"}
    
    for regime in sorted(regimes.unique()):
        regime_res = conditional_results[conditional_results['regime'] == regime]
        sig_horizons = regime_res[regime_res['mean_reverting']]['horizon'].tolist()
        
        print(f"\n{regime_names.get(regime, f'Regime {regime}')}:")
        if sig_horizons:
            print(f"  Mean reversion detected in {len(sig_horizons)}/{len(regime_res)} horizons")
            print(f"  Horizons: {sig_horizons}")
            
            # Half-life at 5-day horizon
            hl_5d = regime_res[regime_res['horizon'] == 5]['half_life'].values
            if len(hl_5d) > 0 and not pd.isna(hl_5d[0]):
                print(f"  Half-life (5-day): {hl_5d[0]:.2f} days")
        else:
            print("  No significant mean reversion detected")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HMM-based credit spread analysis pipeline"
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download fresh data from FRED (default: use cached)"
    )
    
    args = parser.parse_args()
    
    try:
        main(fetch_fresh_data=args.fetch)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
