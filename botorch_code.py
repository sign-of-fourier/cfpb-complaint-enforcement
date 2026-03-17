"""
BoTorch code extracted from CFPB Enforcement Predictor POC (run_pipeline.py)

This file contains the Bayesian Optimization components:
- Search space definition
- Config ↔ parameter vector conversion  
- Sobol quasi-random initialization
- MixedSingleTaskGP surrogate + LogExpectedImprovement acquisition
- The full BO loop (8 Sobol + 40 BO iterations)
- Random search baseline for comparison
"""

import numpy as np
import torch
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf_mixed
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


# ===========================================================================
# SEARCH SPACE DEFINITION
# ===========================================================================
#
# 8 dimensions total:
#   [0] lookback_days      — continuous [90, 730]
#   [1] min_complaints     — discrete [5, 100]
#   [2] class_weight_ratio — continuous [1.0, 20.0]
#   [3] threshold          — continuous [0.1, 0.9]
#   [4] feature_subset     — categorical {0: volume, 1: +dist, 2: +resp, 3: all}
#   [5] model_type         — categorical {0: logistic_regression, 1: random_forest, 2: GBT}
#   [6] text_features      — categorical {0: none, 1: basic}
#   [7] control_match_ratio— discrete {1, 2, 3, 5}
#
# Categorical dimension indices (for MixedSingleTaskGP):
CD = [4, 5, 6, 7]

# Full bounds across all 8 dims (continuous bounds for GP; categoricals are
# enumerated via fixed_features_list during acquisition optimization):
FB = torch.tensor([
    [90.,  5.,  1., 0.1, 0., 0., 0., 1.],   # lower
    [730., 100., 20., 0.9, 3., 2., 1., 5.]   # upper
]).double()

# Enumerate all categorical combos for optimize_acqf_mixed:
# feature_subset ∈ {0,1,2,3}, model_type ∈ {0,1,2}, text_features ∈ {0,1}, control_match_ratio ∈ {1,2,3,5}
ffl = [
    {4: a, 5: b, 6: c, 7: d}
    for a in range(4)
    for b in range(3)
    for c in range(2)
    for d in [1, 2, 3, 5]
]
# Total categorical combos: 4 × 3 × 2 × 4 = 96


# ===========================================================================
# CONFIG ↔ PARAMETER VECTOR CONVERSIONS
# ===========================================================================

MT = {0: 'logistic_regression', 1: 'random_forest', 2: 'gradient_boosted_trees'}

def params_to_config(p):
    """Convert a torch parameter vector (8-dim) to a readable config dict."""
    return {
        'lookback_days':      float(p[0]),
        'min_complaints':     int(p[1]),
        'class_weight_ratio': float(p[2]),
        'threshold':          float(p[3]),
        'feature_subset':     int(p[4]),
        'model_type':         int(p[5]),
        'text_features':      int(p[6]),
        'control_match_ratio':int(p[7]),
    }

def config_to_params(c):
    """Convert a config dict to a torch double tensor (8-dim)."""
    return torch.tensor([
        c['lookback_days'],
        c['min_complaints'],
        c['class_weight_ratio'],
        c['threshold'],
        c['feature_subset'],
        c['model_type'],
        c['text_features'],
        c['control_match_ratio'],
    ]).double()


# ===========================================================================
# SOBOL QUASI-RANDOM INITIALIZATION
# ===========================================================================

def generate_sobol_configs(n, seed=42):
    """
    Generate n quasi-random starting configs using a Sobol sequence
    for the 4 continuous dims, with random categoricals.
    """
    from torch.quasirandom import SobolEngine
    se = SobolEngine(4, scramble=True, seed=seed)
    pts = se.draw(n).double()
    np.random.seed(seed)
    
    configs = []
    for i in range(n):
        configs.append({
            'lookback_days':      float(pts[i, 0] * 640 + 90),    # [90, 730]
            'min_complaints':     int(pts[i, 1] * 95 + 5),        # [5, 100]
            'class_weight_ratio': float(pts[i, 2] * 19 + 1),      # [1, 20]
            'threshold':          float(pts[i, 3] * 0.8 + 0.1),   # [0.1, 0.9]
            'feature_subset':     int(np.random.choice([0, 1, 2, 3])),
            'model_type':         int(np.random.choice([0, 1, 2])),
            'text_features':      int(np.random.choice([0, 1])),
            'control_match_ratio':int(np.random.choice([1, 2, 3, 5])),
        })
    return configs


def generate_random_config():
    """Generate a single uniformly random config."""
    return {
        'lookback_days':      float(np.random.uniform(90, 730)),
        'min_complaints':     int(np.random.uniform(5, 100)),
        'class_weight_ratio': float(np.random.uniform(1, 20)),
        'threshold':          float(np.random.uniform(0.1, 0.9)),
        'feature_subset':     int(np.random.choice([0, 1, 2, 3])),
        'model_type':         int(np.random.choice([0, 1, 2])),
        'text_features':      int(np.random.choice([0, 1])),
        'control_match_ratio':int(np.random.choice([1, 2, 3, 5])),
    }


def clamp_config(cfg):
    """Clamp config values to valid ranges after BO proposal."""
    cfg['lookback_days']      = np.clip(cfg['lookback_days'], 90, 730)
    cfg['min_complaints']     = int(np.clip(cfg['min_complaints'], 5, 100))
    cfg['class_weight_ratio'] = np.clip(cfg['class_weight_ratio'], 1, 20)
    cfg['threshold']          = np.clip(cfg['threshold'], 0.1, 0.9)
    cfg['feature_subset']     = int(np.clip(cfg['feature_subset'], 0, 3))
    cfg['model_type']         = int(np.clip(cfg['model_type'], 0, 2))
    cfg['text_features']      = int(np.clip(cfg['text_features'], 0, 1))
    cfg['control_match_ratio']= int(np.clip(cfg['control_match_ratio'], 1, 5))
    return cfg


# ===========================================================================
# THE BO LOOP
# ===========================================================================

def run_bo_loop(evaluate_fn, n_sobol=8, n_bo=40):
    """
    Full BO optimization loop.
    
    Args:
        evaluate_fn: callable(config_dict) -> float (F1 score)
                     The black-box objective. Each call builds a dataset from
                     the config, trains a model, and returns F1 on held-out data.
        n_sobol: number of Sobol quasi-random cold-start evaluations
        n_bo: number of BO-guided evaluations after cold start
    
    Returns:
        configs: list of all evaluated configs
        f1_scores: list of F1 scores for each config
    """
    configs = []
    f1_scores = []
    
    # ---- Phase 1: Sobol cold start ----
    print(f"Sobol cold start ({n_sobol} evals):")
    for i, cfg in enumerate(generate_sobol_configs(n_sobol)):
        f1, meta = evaluate_fn(cfg)
        configs.append(cfg)
        f1_scores.append(f1)
        print(f"  {i+1}/{n_sobol}: F1={f1:.4f}")
    
    # ---- Phase 2: BO iterations ----
    print(f"\nBO iterations ({n_bo} evals):")
    
    for bi in range(n_bo):
        # Build training data for the GP surrogate
        train_X = torch.stack([config_to_params(c) for c in configs])
        train_Y = torch.tensor(f1_scores).double().unsqueeze(-1)
        
        try:
            # Fit MixedSingleTaskGP (handles continuous + categorical dims)
            gp = MixedSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                cat_dims=CD           # dims [4,5,6,7] are categorical
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)     # optimize GP hyperparameters
            
            # LogExpectedImprovement acquisition function
            acq = LogExpectedImprovement(
                model=gp,
                best_f=train_Y.max()  # current best observed F1
            )
            
            # Optimize acquisition over mixed search space
            # fixed_features_list enumerates all 96 categorical combos;
            # for each combo, optimize the continuous dims
            candidates, _ = optimize_acqf_mixed(
                acq_function=acq,
                bounds=FB,                       # [2, 8] bounds tensor
                fixed_features_list=ffl,         # 96 categorical combos
                q=1,                             # batch size = 1
                num_restarts=5,                  # multi-start optimization
                raw_samples=64,                  # initial random candidates
            )
            
            # Convert BO proposal to config dict
            cfg = params_to_config(candidates.squeeze())
            
        except Exception as e:
            # GP fitting can fail on degenerate data; fall back to random
            print(f"  GP failed ({type(e).__name__}: {e}), random fallback")
            cfg = generate_random_config()
        
        # Clamp to valid ranges
        cfg = clamp_config(cfg)
        
        # Evaluate the proposed config
        f1, meta = evaluate_fn(cfg)
        configs.append(cfg)
        f1_scores.append(f1)
        
        print(f"  BO {bi+1}/{n_bo}: F1={f1:.4f} | best so far={max(f1_scores):.4f} | "
              f"model={MT.get(cfg['model_type'],'?')} | "
              f"lookback={cfg['lookback_days']:.0f}d")
    
    return configs, f1_scores


def run_random_baseline(evaluate_fn, n_evals=48):
    """
    Random search baseline with the same evaluation budget.
    """
    np.random.seed(123)
    configs = []
    f1_scores = []
    
    print(f"\nRandom search baseline ({n_evals} evals):")
    for i in range(n_evals):
        cfg = generate_random_config()
        f1, meta = evaluate_fn(cfg)
        configs.append(cfg)
        f1_scores.append(f1)
        print(f"  {i+1}/{n_evals}: F1={f1:.4f} | best={max(f1_scores):.4f}")
    
    return configs, f1_scores


# ===========================================================================
# USAGE (from run_pipeline.py)
# ===========================================================================
#
# The evaluate() function in run_pipeline.py does the following for each config:
#   1. build_dataset(lookback_days, min_complaints, control_match_ratio)
#      - For each enforcement action, pull complaints in the lookback window
#      - Match with control companies of similar size
#   2. compute_features() for selected feature groups
#   3. Train LogReg / RF / GBT with class_weight_ratio
#   4. Predict on 30% held-out, threshold at cfg['threshold']
#   5. Return F1 score
#
# Results from the actual run:
#   BO best:    F1=1.0000 at eval #19 (mean across 48 evals: 0.725)
#   Random best: F1=0.8182 (mean across 48 evals: 0.389)
#   BO improvement: 86% higher mean F1
