"""
src/models/__init__.py
======================
Core quantitative modeling package for Inter IIT Tech Meet 15.0 - Track A.

Modules:
--------
- volatility: High-efficiency volatility estimators (Garman-Klass, Parkinson, EWMA, Realized).
- ranking: Cross-sectional factor ranking (Trend Sharpe, Pullback, Low-Vol) and macro trend timing.
- portfolio_opt: Volatility-adjusted portfolio construction and Risk Parity optimization.
"""

from src.models.volatility import VolatilityModel
from src.models.ranking import CrossSectionalRanker
from src.models.portfolio_opt import VolatilityAdjustedPortfolioOptimizer

__all__ = [
    "VolatilityModel",
    "CrossSectionalRanker",
    "VolatilityAdjustedPortfolioOptimizer",
]
