"""
src/models/volatility.py
========================
Multi-Estimator Volatility Modeling Engine for Quantitative Asset Allocation.
Inter IIT Tech Meet 15.0 Quant Evaluation - Track A: Volatility-Adjusted Cross-Sectional Strategy.

Theoretical Overview:
---------------------
Accurate volatility estimation is the foundational cornerstone of risk-budgeting
and risk parity asset allocation. Standard close-to-close sample volatility discards
all intraday price discovery, suffering from high estimator variance and slow convergence.
This module implements four institutional volatility estimators spanning range-based,
extreme-value, exponential decay, and realized variance paradigms.

Implemented Estimators:
-----------------------
1. Garman-Klass (1980) Range Volatility:
   - Utilizes High, Low, Open, and Close to extract maximum diffusion information.
   - ~7.4x to 8.4x higher statistical efficiency than close-to-close variance.
   - Primary volatility estimator used for cross-sectional ranking and position sizing.

2. Parkinson (1980) High-Low Extreme Value Volatility:
   - Based on the continuous-time distribution of Brownian motion extrema.
   - ~5x higher statistical efficiency than close-to-close variance.

3. Exponentially Weighted Moving Average (EWMA / RiskMetrics):
   - Decaying exponential memory (lambda = 0.94) capturing volatility clustering
     and temporal persistence (Engle 1982; Mandelbrot 1963).

4. Classical Realized Volatility:
   - Unbiased close-to-close sample standard deviation with Bessel correction.

5. Composite Blended Volatility:
   - Ensemble weighting (70% Garman-Klass + 30% EWMA) balancing intraday precision
     with rapid jump responsiveness.

References:
-----------
- Garman, M. B., & Klass, M. J. (1980). On the Estimation of Security Price Volatilities
  from Historical Data. The Journal of Business, 53(1), 67-78.
- Parkinson, M. (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return.
  The Journal of Business, 53(1), 61-65.
- J.P. Morgan / Reuters (1996). RiskMetrics Technical Document (4th ed.).
"""

from __future__ import annotations
import math
from typing import List, Sequence, Optional
import numpy as np


class VolatilityModel:
    """
    Computes rigorous, mathematically consistent volatility estimates across multiple
    academic and practitioner estimators.

    Parameters:
    -----------
    default_window : int
        The default rolling lookback window in trading days (default: 20 trading days ~ 1 month).
    annualization_factor : float
        The annualization constant for daily data (default: 252.0 trading days per calendar year).
    """

    def __init__(self, default_window: int = 20, annualization_factor: float = 252.0) -> None:
        self.default_window = default_window
        self.annualization_factor = annualization_factor

    def garman_klass_volatility(
        self,
        opens: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        window: Optional[int] = None
    ) -> float:
        """
        Garman-Klass (1980) Range Volatility Estimator.

        Mathematical Formulation:
        -------------------------
        Let H_t, L_t, O_t, C_t be the High, Low, Open, and Close prices for trading day t.
        The daily Garman-Klass variance estimator is defined as:
            sigma_GK,t^2 = 0.5 * [ln(H_t / L_t)]^2 - [2 * ln(2) - 1] * [ln(C_t / O_t)]^2

        Over a rolling window of k trading days:
            sigma_GK^2 = (1 / k) * sum_{t=1}^k (sigma_GK,t^2)
            Annualized Volatility = sqrt(252 * sigma_GK^2)

        Efficiency & Rationale:
        -----------------------
        Under geometric Brownian motion with zero drift, Garman and Klass (1980) demonstrated
        that this estimator has a theoretical variance of ~0.17 relative to 1.33 for the
        standard close-to-close estimator, yielding a relative statistical efficiency of ~7.4x.
        It accounts for both the intraday price range (H/L) and opening jumps (C/O).

        Parameters:
        -----------
        opens : Sequence[float]
            Historical opening prices.
        highs : Sequence[float]
            Historical high prices.
        lows : Sequence[float]
            Historical low prices.
        closes : Sequence[float]
            Historical closing prices.
        window : Optional[int]
            Lookback window length. If None, uses self.default_window.

        Returns:
        --------
        float : Annualized Garman-Klass volatility, clamped between [0.05, 2.0] for stability.
        """
        w = window or self.default_window
        n = len(closes)
        if n < 2:
            return 0.25  # Neutral empirical default prior when history is insufficient

        k = min(n, w)
        o = np.array(opens[-k:], dtype=np.float64)
        h = np.array(highs[-k:], dtype=np.float64)
        l = np.array(lows[-k:], dtype=np.float64)
        c = np.array(closes[-k:], dtype=np.float64)

        # Numerical safety: prevent non-positive inputs to log
        o = np.maximum(o, 1e-4)
        h = np.maximum(h, 1e-4)
        l = np.maximum(l, 1e-4)
        c = np.maximum(c, 1e-4)

        # Ensure high >= low to prevent non-physical log ratios
        h = np.maximum(h, l)

        # Compute log terms
        log_hl = np.log(h / l) ** 2
        log_co = np.log(c / o) ** 2

        # Daily variance components
        var_sample = 0.5 * log_hl - (2.0 * math.log(2.0) - 1.0) * log_co

        # Mean sample variance over the lookback window
        mean_var = float(np.mean(var_sample))

        # Annualize and clamp to economically reasonable volatility bounds
        annualized_vol = math.sqrt(max(1e-8, mean_var * self.annualization_factor))
        return float(np.clip(annualized_vol, 0.05, 2.0))

    def parkinson_volatility(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        window: Optional[int] = None
    ) -> float:
        """
        Parkinson (1980) High-Low Extreme Value Volatility Estimator.

        Mathematical Formulation:
        -------------------------
        Let H_t and L_t denote the High and Low prices for day t.
            sigma_P,t^2 = [ln(H_t / L_t)]^2 / [4 * ln(2)]

        Over a rolling window of k trading days:
            sigma_P^2 = (1 / k) * sum_{t=1}^k (sigma_P,t^2)
            Annualized Volatility = sqrt(252 * sigma_P^2)

        Efficiency & Rationale:
        -----------------------
        Derived from the distribution of the range of continuous-time Brownian motion.
        Approximately 5 times more efficient than close-to-close volatility, though it assumes
        zero opening jump and continuous trading without drift.

        Parameters:
        -----------
        highs : Sequence[float]
            Historical high prices.
        lows : Sequence[float]
            Historical low prices.
        window : Optional[int]
            Lookback window length. If None, uses self.default_window.

        Returns:
        --------
        float : Annualized Parkinson volatility, clamped between [0.05, 2.0].
        """
        w = window or self.default_window
        n = len(highs)
        if n < 2:
            return 0.25

        k = min(n, w)
        h = np.array(highs[-k:], dtype=np.float64)
        l = np.array(lows[-k:], dtype=np.float64)
        h = np.maximum(h, 1e-4)
        l = np.maximum(l, 1e-4)
        h = np.maximum(h, l)

        var_sample = (np.log(h / l) ** 2) / (4.0 * math.log(2.0))
        mean_var = float(np.mean(var_sample))
        annualized_vol = math.sqrt(max(1e-8, mean_var * self.annualization_factor))
        return float(np.clip(annualized_vol, 0.05, 2.0))

    def ewma_volatility(
        self,
        closes: Sequence[float],
        decay_factor: float = 0.94
    ) -> float:
        """
        Exponentially Weighted Moving Average (EWMA) Volatility Estimator.

        Mathematical Formulation:
        -------------------------
        Let r_t = (C_t / C_{t-1}) - 1 be the simple percentage return on day t.
        The recursive variance update is defined by RiskMetrics (1996):
            sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_t^2

        With standard daily decay parameter lambda = 0.94:
            - Effective memory half-life = ln(0.5) / ln(0.94) ~ 11.2 trading days.
            - Responds rapidly to volatility spikes and tail risk events.

        Parameters:
        -----------
        closes : Sequence[float]
            Historical closing prices.
        decay_factor : float
            Smoothing parameter lambda in [0, 1] (default: 0.94).

        Returns:
        --------
        float : Annualized EWMA volatility, clamped between [0.05, 2.0].
        """
        n = len(closes)
        if n < 3:
            return 0.25

        returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, n)]
        var = returns[0] ** 2
        for r in returns[1:]:
            var = decay_factor * var + (1.0 - decay_factor) * (r ** 2)

        annualized_vol = math.sqrt(max(1e-8, var * self.annualization_factor))
        return float(np.clip(annualized_vol, 0.05, 2.0))

    def realized_volatility(
        self,
        closes: Sequence[float],
        window: Optional[int] = None
    ) -> float:
        """
        Classical Close-to-Close Realized Volatility Estimator.

        Mathematical Formulation:
        -------------------------
        Let r_t = (C_t / C_{t-1}) - 1.
            s = sqrt( (1 / (k - 1)) * sum_{t=1}^k (r_t - mean(r))^2 )
            Annualized Volatility = s * sqrt(252)

        Provides an unbiased benchmark baseline across all traditional financial literature.

        Parameters:
        -----------
        closes : Sequence[float]
            Historical closing prices.
        window : Optional[int]
            Lookback window length. If None, uses self.default_window.

        Returns:
        --------
        float : Annualized realized volatility, clamped between [0.05, 2.0].
        """
        w = window or self.default_window
        n = len(closes)
        if n < 3:
            return 0.25

        k = min(n, w)
        c = np.array(closes[-k:], dtype=np.float64)
        returns = np.diff(c) / c[:-1]
        std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.015
        annualized_vol = std_ret * math.sqrt(self.annualization_factor)
        return float(np.clip(annualized_vol, 0.05, 2.0))

    def estimate_composite_volatility(
        self,
        opens: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        window: Optional[int] = None
    ) -> float:
        """
        Composite Multi-Scale Volatility Estimator.

        Ensemble Formulation:
        ---------------------
        sigma_composite = 0.70 * sigma_GarmanKlass + 0.30 * sigma_EWMA

        Combines the high statistical efficiency of Garman-Klass intraday range modeling
        with the rapid exponential jump adaptation of EWMA.

        Parameters:
        -----------
        opens : Sequence[float]
            Historical opening prices.
        highs : Sequence[float]
            Historical high prices.
        lows : Sequence[float]
            Historical low prices.
        closes : Sequence[float]
            Historical closing prices.
        window : Optional[int]
            Lookback window length.

        Returns:
        --------
        float : Annualized composite volatility estimate in [0.08, 1.50].
        """
        gk = self.garman_klass_volatility(opens, highs, lows, closes, window)
        ewma = self.ewma_volatility(closes)
        blended = 0.70 * gk + 0.30 * ewma
        return float(np.clip(blended, 0.08, 1.50))
