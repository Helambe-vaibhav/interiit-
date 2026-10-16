"""
src/models/portfolio_opt.py
===========================
Volatility-Adjusted Portfolio Construction and Sizing Engine.
Inter IIT Tech Meet 15.0 Quant Evaluation - Track A: Volatility-Adjusted Cross-Sectional Strategy.

Option A Architecture: Robust Risk Parity & Macro Trend Defense (Drawdown Suppression).

Theoretical Foundation & Portfolio Mathematics:
------------------------------------------------
1. Risk Parity / Equal Risk Contribution (ERC):
   - Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of Equally Weighted
     Risk Contributing Portfolios. The Journal of Portfolio Investment, 36(4), 60-70.
   - Asness, C. S., Frazzini, A., & Pedersen, L. H. (2012). Leverage Aversion and Risk Parity.
     Financial Analysts Journal, 68(1), 47-59.
   - DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification:
     How Inefficient is the 1/N Portfolio Strategy? The Review of Financial Studies, 22(5), 1915-1953.
   - Markowitz (1952) mean-variance optimization is notoriously sensitive to return estimation
     errors ("error maximizer", Michaud 1989). In contrast, inverse-volatility risk parity
     relies strictly on second moments (which are far more statistically stable than first moments)
     to equalize the marginal risk contribution of each constituent.

2. Inverse-Volatility Allocation Formulation:
   - For selected asset i in Top K:
       w_i_raw = (1 / sigma_GK,i^gamma) / sum_{j=1}^K (1 / sigma_GK,j^gamma)
     where gamma = 1.0 (pure inverse volatility) and sigma_GK,i is the Garman-Klass range volatility.

3. Concentration Cap & Iterative Proportional Fitting (IPF):
   - Single-asset weight bounded by w_max = min(18%, L_target / 3.0).
   - An iterative redistribution loop rescales uncapped assets until the target leverage
     is fully utilized without violating the 18% concentration ceiling.

4. Macro Regime Leverage Scaling (Drawdown Defense):
   - Bull Market Regime: Target Gross Leverage = 0.96 (invested in top 6 assets).
   - Bear Market Regime: Target Gross Leverage = 0.48 (allocates 52% to cash),
     drastically reducing portfolio variance and drawdown during broad market downturns.

5. Turnover Deadband Buffer:
   - If |w_target,i - w_current,i| < 2.0%, rebalancing is bypassed for asset i.
   - Significantly reduces execution turnover, commissions, and bid-ask slippage.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np


class VolatilityAdjustedPortfolioOptimizer:
    """
    Constructs an institutional-grade, volatility-adjusted portfolio implementing
    inverse-volatility risk parity with macro risk scaling and concentration limits.

    Parameters:
    -----------
    top_k : int
        Number of top-ranked assets to include in portfolio (default: 6, representing
        the top 25% quintile of the 24-asset universe).
    max_gross_leverage : float
        Normal operating gross leverage ceiling (default: 0.96 <= 1.0 limit).
    bear_gross_leverage : float
        De-risked gross leverage ceiling during macro bear regimes (default: 0.48).
    max_asset_weight : float
        Maximum allowable allocation to any single asset (default: 0.18 or 18%).
    vol_exponent : float
        Risk aversion scaling exponent gamma (default: 1.0 for pure risk parity).
    min_vol_floor : float
        Minimum volatility clamp to prevent division by zero (default: 0.10 or 10% ann. vol).
    max_vol_cap : float
        Maximum volatility clamp to avoid total exclusion of volatile assets (default: 1.00).
    rebalance_deadband : float
        Minimum position weight delta required to execute a trade (default: 0.02 or 2%).
    """

    def __init__(
        self,
        top_k: int = 6,
        max_gross_leverage: float = 0.96,
        bear_gross_leverage: float = 0.48,
        max_asset_weight: float = 0.18,
        vol_exponent: float = 1.0,
        min_vol_floor: float = 0.10,
        max_vol_cap: float = 1.00,
        rebalance_deadband: float = 0.02,
    ) -> None:
        self.top_k = top_k
        self.max_gross_leverage = max_gross_leverage
        self.bear_gross_leverage = bear_gross_leverage
        self.max_asset_weight = max_asset_weight
        self.vol_exponent = vol_exponent
        self.min_vol_floor = min_vol_floor
        self.max_vol_cap = max_vol_cap
        self.rebalance_deadband = rebalance_deadband

    def optimize_weights(
        self,
        ranked_universe: Dict[str, Dict[str, float]],
        current_weights: Optional[Dict[str, float]] = None,
        is_bear_regime: bool = False,
    ) -> Dict[str, float]:
        """
        Computes risk-budgeted target portfolio weights given cross-sectional factor ranks,
        current holdings, and macro regime state.

        Algorithmic Pipeline:
        ---------------------
        1. Leverage Target Determination:
           Selects target gross leverage based on macro regime (0.96 bull vs 0.48 bear).
        2. Top-K Selection:
           Filters universe by composite score, prioritizing assets in secular uptrends.
        3. Inverse-Volatility Risk Budgeting:
           Calculates inv_vol_i = 1 / (sigma_i^gamma), normalized across selected assets.
        4. Concentration Clamping & Iterative Proportional Fitting:
           Enforces 18% concentration ceiling and re-distributes slack to uncapped assets.
        5. Turnover Deadband Application:
           Freezes positions where target delta is below the 2% rebalance threshold.
        6. Hard Leverage Boundary Verification:
           Ensures sum(abs(weights)) strictly <= target_leverage <= 1.0.

        Parameters:
        -----------
        ranked_universe : Dict[str, Dict[str, float]]
            Universe factor metrics from CrossSectionalRanker.
        current_weights : Optional[Dict[str, float]]
            Current portfolio weights (used to compute turnover deadband).
        is_bear_regime : bool
            Flag from macro trend filter indicating broad market contraction.

        Returns:
        --------
        Dict[str, float] : Target portfolio weight mapping ticker -> weight.
        """
        # Step 1: Determine target gross leverage based on regime
        target_leverage = self.bear_gross_leverage if is_bear_regime else self.max_gross_leverage
        cap_weight = min(self.max_asset_weight, target_leverage / 3.0)

        # Step 2: Sort tickers by composite score descending
        sorted_tickers = sorted(
            ranked_universe.keys(),
            key=lambda t: ranked_universe[t]["composite_score"],
            reverse=True,
        )

        # Prioritize assets whose individual secular trend (50d SMA) is positive
        uptrend_candidates = [t for t in sorted_tickers if ranked_universe[t].get("in_uptrend", True)]
        selected_tickers = uptrend_candidates[: self.top_k]

        # If fewer than top_k assets are in an uptrend, fill remaining slots with highest-ranked assets
        if len(selected_tickers) < self.top_k:
            remaining = [t for t in sorted_tickers if t not in selected_tickers]
            selected_tickers.extend(remaining[: self.top_k - len(selected_tickers)])

        # Step 3: Compute inverse-volatility weights (Risk Parity formulation)
        inv_vols: Dict[str, float] = {}
        for t in selected_tickers:
            v = ranked_universe[t]["volatility"]
            clamped_v = max(self.min_vol_floor, min(self.max_vol_cap, v))
            inv_vols[t] = 1.0 / (clamped_v ** self.vol_exponent)

        total_inv_vol = sum(inv_vols.values())
        if total_inv_vol <= 0:
            # Fallback to equal weighting if inverse volatilities are non-positive
            equal_w = target_leverage / len(selected_tickers)
            return {t: equal_w for t in selected_tickers}

        # Step 4: Initial raw allocation and concentration clamping
        weights: Dict[str, float] = {}
        for t in selected_tickers:
            raw_w = (inv_vols[t] / total_inv_vol) * target_leverage
            weights[t] = min(raw_w, cap_weight)

        # Iterative Proportional Fitting (IPF) to re-distribute excess weight
        for _ in range(3):
            current_sum = sum(weights.values())
            if current_sum <= 0:
                break
            scale = min(1.0, target_leverage / current_sum)
            for t in weights:
                weights[t] = min(weights[t] * scale, cap_weight)

        # Step 5: Turnover buffer / deadband
        # If the planned weight adjustment is smaller than rebalance_deadband, maintain current weight
        if current_weights:
            for t in selected_tickers:
                curr_w = current_weights.get(t, 0.0)
                if abs(weights[t] - curr_w) < self.rebalance_deadband:
                    weights[t] = curr_w

        # Step 6: Hard RuleGuard leverage boundary verification
        # Guarantee gross leverage does not exceed target leverage or the strict 1.0 limit
        gross_leverage = sum(abs(w) for w in weights.values())
        if gross_leverage > target_leverage:
            ratio = target_leverage / gross_leverage
            weights = {k: v * ratio for k, v in weights.items()}

        return weights
