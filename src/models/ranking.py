"""
src/models/ranking.py
=====================
Cross-Sectional Factor Ranking and Macro Regime Detection Engine.
Inter IIT Tech Meet 15.0 Quant Evaluation - Track A: Volatility-Adjusted Cross-Sectional Strategy.

Option A Architecture: Robust, Diversified, Multi-Regime Academic Pillar Model
with Classical 100-day Macro Trend Timing (Faber 2007) for Drawdown Suppression.

Theoretical Foundation & Academic Literature:
---------------------------------------------
1. Quality Momentum / Trend Sharpe (Pillar 1):
   - Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers:
     Implications for Stock Market Efficiency. The Journal of Finance, 48(1), 65-91.
   - Daniel, K., & Moskowitz, T. J. (2016). Momentum Crashes. Journal of Financial Economics.
   - Formulation: 60-day cumulative return normalized by 20-day Garman-Klass volatility.
   - Rationale: High raw momentum achieved through erratic, volatile spikes is penalized;
     steady, high-information-ratio compounders are prioritized.

2. Short-Term Pullback / Liquidity Reversion (Pillar 2):
   - Lehmann, B. N. (1990). Fads, Martingales, and Market Efficiency. QJE, 105(1), 1-28.
   - Lo, A. W., & MacKinlay, A. C. (1990). When are Contrarian Profits Due to Stock Market
     Overreaction? The Review of Financial Studies, 3(2), 175-205.
   - Formulation: Negative standardized distance to the 10-day simple moving average.
   - Rationale: Prevents the strategy from buying at local overextended peaks; enters
     high-momentum trends on temporary liquidity dips.

3. Low-Volatility Anomaly Factor (Pillar 3):
   - Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). The Cross-Section of Volatility
     and Expected Returns. The Journal of Finance, 61(1), 259-299.
   - Frazzini, A., & Pedersen, L. H. (2014). Betting Against Beta. JFE, 111(1), 1-25.
   - Formulation: Inverse Garman-Klass volatility (-sigma_GK).
   - Rationale: Leveraged institutional constraints create structural mispricing, allowing
     lower-volatility equities to generate superior risk-adjusted alpha.

4. Macro Market Trend Defense (Drawdown Mitigation):
   - Faber, M. T. (2007). A Quantitative Approach to Tactical Asset Allocation.
     The Journal of Wealth Management, 9(4), 69-79.
   - Formulation: 100-day simple moving average of the equal-weighted 24-asset universe index.
   - Rationale: Flags severe macro market contractions and de-risks portfolio exposure,
     slashing maximum drawdown during market crashes (e.g., 2018, 2020) without curve-fitting.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np

from src.models.volatility import VolatilityModel


class CrossSectionalRanker:
    """
    Ranks a multi-asset universe across three balanced academic pillars and evaluates
    macro market regimes for dynamic risk scaling.

    Parameters:
    -----------
    vol_model : VolatilityModel
        Instantiated volatility model providing Garman-Klass range calculations.
    mom_window : int
        Lookback window in trading days for medium-term momentum (default: 60 days ~ 3 months).
    sma_short : int
        Lookback window for short-term liquidity pullback (default: 10 days ~ 2 weeks).
    sma_trend : int
        Lookback window for individual asset secular trend filter (default: 50 days).
    sma_macro : int
        Lookback window for broad market macro regime filter (default: 100 days).
    """

    def __init__(
        self,
        vol_model: VolatilityModel,
        mom_window: int = 60,
        sma_short: int = 10,
        sma_trend: int = 50,
        sma_macro: int = 100,
    ) -> None:
        self.vol_model = vol_model
        self.mom_window = mom_window
        self.sma_short = sma_short
        self.sma_trend = sma_trend
        self.sma_macro = sma_macro

    def compute_asset_factors(
        self,
        ticker: str,
        history: Dict[str, List[float]],
    ) -> Tuple[float, float, float, float, bool]:
        """
        Computes the three primary alpha factors and trend status for an individual asset.

        Mathematical Definitions:
        -------------------------
        1. Trend Sharpe:
           R_60 = (Close_t - Close_{t-60}) / Close_{t-60}
           Trend Sharpe = R_60 / (sigma_GK + 1e-4)

        2. Short-Term Pullback (Distance to 10d SMA):
           SMA_10 = mean(Close_{t-9:t})
           Std_10 = std(Close_{t-9:t})
           Pullback = - (Close_t - SMA_10) / (Std_10 + 1e-4)
           (Positive when price is below SMA_10, signaling a temporary dip)

        3. Low-Volatility Factor:
           LowVol = - sigma_GK

        4. Secular Uptrend Indicator:
           SMA_50 = mean(Close_{t-49:t})
           InUptrend = Close_t > SMA_50

        Parameters:
        -----------
        ticker : str
            Asset identifier.
        history : Dict[str, List[float]]
            Dictionary mapping f"{ticker}_{field}" to rolling price lists.

        Returns:
        --------
        Tuple[float, float, float, float, bool]:
            (trend_sharpe, dist_sma, inv_vol, raw_vol, in_uptrend)
        """
        closes = history[f"{ticker}_close"]
        highs = history[f"{ticker}_high"]
        lows = history[f"{ticker}_low"]
        opens = history[f"{ticker}_open"]

        n = len(closes)
        # Ensure sufficient history for 60-day momentum
        if n < self.mom_window + 1:
            return 0.0, 0.0, 0.0, 0.25, False

        p_curr = closes[-1]

        # -------------------------------------------------------------
        # Pillar 1: Quality Momentum / Trend Sharpe (Jegadeesh & Titman 1993)
        # -------------------------------------------------------------
        p_past = closes[-self.mom_window - 1]
        mom = (p_curr - p_past) / p_past if p_past > 0 else 0.0

        # Estimate 20-day annualized Garman-Klass volatility
        vol = self.vol_model.garman_klass_volatility(
            opens=opens, highs=highs, lows=lows, closes=closes, window=20
        )
        trend_sharpe = mom / (vol + 1e-4)

        # -------------------------------------------------------------
        # Pillar 2: Short-Term Pullback / Liquidity Reversion (Lehmann 1990)
        # -------------------------------------------------------------
        k_short = min(n, self.sma_short)
        short_vals = closes[-k_short:]
        sma10 = float(np.mean(short_vals))
        std10 = float(np.std(short_vals)) if len(short_vals) > 1 else 1.0
        # Negative sign: rewards prices that are temporarily below recent average
        dist_sma = -(p_curr - sma10) / (std10 + 1e-4)

        # -------------------------------------------------------------
        # Pillar 3: Low-Volatility Anomaly Factor (Ang et al. 2006)
        # -------------------------------------------------------------
        # Assets with lower volatility receive a higher positive rank
        inv_vol = -vol

        # -------------------------------------------------------------
        # Asset Secular Trend Filter (50-day SMA)
        # -------------------------------------------------------------
        k_trend = min(n, self.sma_trend)
        sma50 = float(np.mean(closes[-k_trend:]))
        in_uptrend = p_curr > sma50

        return float(trend_sharpe), float(dist_sma), float(inv_vol), float(vol), in_uptrend

    def rank_universe(
        self,
        tickers: List[str],
        history: Dict[str, List[float]],
    ) -> Tuple[Dict[str, Dict[str, float]], bool]:
        """
        Performs cross-sectional factor ranking across the entire asset universe
        and computes macro regime status using broad market trend timing.

        Cross-Sectional Standardization:
        --------------------------------
        For each factor f across universe tickers i = 1, ..., N:
            Z_i = clip( (f_i - mean(f)) / std(f), -3.0, 3.0 )
        Winsorizing at +/- 3.0 standard deviations prevents individual outlier
        assets from dominating portfolio selection.

        Multi-Factor Composite Score:
        -----------------------------
        Equal 1/3 weighting across the three academic pillars:
            Score_i = (1/3) * Z(Trend Sharpe)_i
                    + (1/3) * Z(Pullback)_i
                    + (1/3) * Z(LowVol)_i
                    + (0.25 if InUptrend_i else 0.0)

        Equal weighting is purposefully chosen to prevent in-sample parameter
        curve-fitting, ensuring true out-of-sample robustness.

        Macro Regime Timing (Faber 2007):
        ---------------------------------
        Equal-weighted market index M_t = (1 / N) * sum_{i=1}^N P_{t,i}.
        If M_t < SMA_100(M), flags is_bear_regime = True.

        Parameters:
        -----------
        tickers : List[str]
            List of universe asset tickers.
        history : Dict[str, List[float]]
            Rolling price series dictionary.

        Returns:
        --------
        Tuple[Dict[str, Dict[str, float]], bool]:
            - Dict mapping ticker -> {composite_score, volatility, in_uptrend}
            - is_bear_regime boolean indicating market contraction state.
        """
        raw_sharpe: Dict[str, float] = {}
        raw_dist: Dict[str, float] = {}
        raw_lowvol: Dict[str, float] = {}
        vols: Dict[str, float] = {}
        in_uptrends: Dict[str, bool] = {}

        # 1. Compute raw factor values across all universe assets
        for t in tickers:
            s, d, lv, v, up = self.compute_asset_factors(t, history)
            raw_sharpe[t] = s
            raw_dist[t] = d
            raw_lowvol[t] = lv
            vols[t] = v
            in_uptrends[t] = up

        # 2. Cross-sectional Z-score standardization with Winsorization
        def z_score(d: Dict[str, float]) -> Dict[str, float]:
            vals = np.array(list(d.values()), dtype=np.float64)
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
            if sigma < 1e-6:
                return {k: 0.0 for k in d}
            return {k: float(np.clip((v - mu) / sigma, -3.0, 3.0)) for k, v in d.items()}

        z_sharpe = z_score(raw_sharpe)
        z_dist = z_score(raw_dist)
        z_lowvol = z_score(raw_lowvol)

        # 3. Construct equal-weighted multi-factor composite score
        output: Dict[str, Dict[str, float]] = {}
        for t in tickers:
            comp_score = (
                (1.0 / 3.0) * z_sharpe[t]
                + (1.0 / 3.0) * z_dist[t]
                + (1.0 / 3.0) * z_lowvol[t]
            )
            # Secular trend bonus: favor assets trading above 50d SMA
            if in_uptrends[t]:
                comp_score += 0.25

            output[t] = {
                "composite_score": comp_score,
                "volatility": vols[t],
                "in_uptrend": in_uptrends[t],
            }

        # 4. Evaluate broad macro market regime (100-day trend of equal-weighted universe)
        first_t = tickers[0]
        n_hist = len(history.get(f"{first_t}_close", []))
        is_bear_regime = False

        if n_hist >= self.sma_macro:
            # Construct synthetic equal-weighted market index series over past 100 bars
            mkt_prices = np.zeros(self.sma_macro, dtype=np.float64)
            for t in tickers:
                mkt_prices += np.array(history[f"{t}_close"][-self.sma_macro:], dtype=np.float64)
            mkt_prices /= len(tickers)
            mkt_sma = float(np.mean(mkt_prices))
            current_mkt = mkt_prices[-1]
            # Flag bear regime if current broad market is below its 100-day moving average
            is_bear_regime = current_mkt < mkt_sma

        return output, is_bear_regime
