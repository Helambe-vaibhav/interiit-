"""
src/research/overfitting_analysis.py
Rigorous Overfitting Diagnosis & Walk-Forward Cross-Validation for Track A.
Compares:
1. Concentrated Model (Top 3, 38% cap, tuned parameters) - High Overfitting Risk
2. Robust Diversified Model (Top 6, 18% cap, Risk Parity, Volatility Targeting, Academic Factors)
"""

import numpy as np
import pandas as pd
from scipy import stats

def analyze_overfitting():
    print("=" * 70)
    print("OVERFITTING & GENERALIZATION AUDIT: WALK-FORWARD REGIME SPLITS")
    print("=" * 70)

    df = pd.read_csv("data/Dataset_PS-A.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'ticker']).reset_index(drop=True)

    close_p = df.pivot(index='date', columns='ticker', values='close')
    open_p = df.pivot(index='date', columns='ticker', values='open')
    high_p = df.pivot(index='date', columns='ticker', values='high')
    low_p = df.pivot(index='date', columns='ticker', values='low')

    # Garman-Klass Volatility
    log_hl = np.log(high_p / low_p) ** 2
    log_co = np.log(close_p / open_p) ** 2
    gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    gk_vol20 = np.sqrt(np.maximum(1e-8, gk_var.rolling(20).mean() * 252))

    # Factor signals for Robust Model:
    # 1. Quality Momentum (60d return / 20d GK vol)
    mom60 = close_p.pct_change(60)
    trend_sharpe = mom60 / (gk_vol20 + 1e-4)
    # 2. Short-term Pullback (10d SMA)
    sma10 = close_p.rolling(10).mean()
    dist10 = -(close_p - sma10) / (close_p.rolling(10).std() + 1e-4)
    # 3. Low Volatility Anomaly Factor
    inv_vol = -gk_vol20
    # 4. Long-term trend filter (SMA 100)
    sma100 = close_p.rolling(100).mean()
    mkt_index = close_p.mean(axis=1)
    mkt_sma100 = mkt_index.rolling(100).mean()

    def cross_z(f):
        return f.apply(lambda r: (r - r.mean()) / (r.std() + 1e-6), axis=1)

    z_sharpe = cross_z(trend_sharpe)
    z_dist = cross_z(dist10)
    z_lowvol = cross_z(inv_vol)

    # Robust composite: Equal-weighted 3 academic pillars
    robust_composite = (1/3) * z_sharpe + (1/3) * z_dist + (1/3) * z_lowvol

    # Define 3 distinct macroeconomic regimes:
    # Regime 1: 2017-2018 (Steady Bull)
    # Regime 2: 2019-2020 (COVID Shock & V-Recovery)
    # Regime 3: 2021-2022 (Inflation, Rate Hikes & Tech Consolidation)
    regimes = {
        'Regime 1 (2017-2018 Steady Bull)': ('2017-01-01', '2018-12-31'),
        'Regime 2 (2019-2020 COVID & Crash)': ('2019-01-01', '2020-12-31'),
        'Regime 3 (2021-2022 Rate Hikes & Bear)': ('2021-01-01', '2022-09-01'),
        'Full History (2017-2022)': ('2017-01-01', '2022-09-01')
    }

    models = {
        'Concentrated (Top 3, 38% cap, Tuned)': {
            'top_k': 3,
            'max_w': 0.38,
            'vol_exp': 0.5,
            'signal': 0.60 * z_sharpe + 0.40 * z_dist + 0.60 * (close_p > close_p.rolling(50).mean()).astype(float),
            'vol_target': False
        },
        'Robust Diversified (Top 6, 18% cap, Risk Parity, Vol Targeting)': {
            'top_k': 6,
            'max_w': 0.18,
            'vol_exp': 1.0,  # Pure inverse-vol risk parity
            'signal': robust_composite,
            'vol_target': True # De-risk during market downturns
        }
    }

    results = []

    for m_name, m_cfg in models.items():
        top_k = m_cfg['top_k']
        max_w = m_cfg['max_w']
        vol_exp = m_cfg['vol_exp']
        composite = m_cfg['signal']
        use_vol_target = m_cfg['vol_target']

        for r_name, (start_dt, end_dt) in regimes.items():
            rebalance_dates = close_p.loc[start_dt:end_dt].index[::5]
            if len(rebalance_dates) < 10:
                continue

            equity = 100000.0
            equity_hist = [equity]
            current_w = {}

            for i in range(len(rebalance_dates) - 1):
                dt = rebalance_dates[i]
                next_dt = rebalance_dates[i + 1]

                score = composite.loc[dt].dropna()
                vol = gk_vol20.loc[dt].dropna()
                common = score.index.intersection(vol.index)
                if len(common) < 12:
                    equity_hist.append(equity)
                    continue

                sorted_t = score[common].sort_values(ascending=False).index[:top_k]

                # Risk Parity weighting (1 / vol)
                inv_v = 1.0 / (np.clip(vol[sorted_t], 0.10, 0.80) ** vol_exp)
                
                # Volatility targeting / Market regime hedge
                target_lev = 0.96
                if use_vol_target:
                    # If market index is below 100d SMA or universe average vol > 32%, scale down leverage
                    m_idx = mkt_index.loc[dt]
                    m_sma = mkt_sma100.loc[dt]
                    avg_v = vol.mean()
                    if m_idx < m_sma or avg_v > 0.32:
                        target_lev = 0.50  # 50% cash preservation during regime shifts

                raw_w = (inv_v / inv_v.sum()) * target_lev
                weights = {t: min(raw_w[t], max_w) for t in sorted_t}
                scale = min(1.0, target_lev / max(1e-6, sum(weights.values())))
                weights = {t: min(w * scale, max_w) for t, w in weights.items()}

                # Return
                p_curr = close_p.loc[dt]
                p_next = close_p.loc[next_dt]
                ret_period = (p_next - p_curr) / p_curr

                port_ret = sum(float(weights.get(t, 0.0)) * ret_period.get(t, 0.0) for t in sorted_t)
                turnover = sum(abs(weights.get(t, 0.0) - current_w.get(t, 0.0)) for t in set(weights) | set(current_w))
                port_ret -= turnover * 0.0001

                equity *= (1.0 + port_ret)
                equity_hist.append(equity)
                current_w = dict(weights)

            eq_series = pd.Series(equity_hist, index=rebalance_dates)
            tot_ret = (equity - 100000.0) / 100000.0 * 100.0
            n_years = (rebalance_dates[-1] - rebalance_dates[0]).days / 365.25
            cagr = ((equity / 100000.0) ** (1.0 / n_years) - 1.0) * 100.0 if n_years > 0 else 0.0
            rets = eq_series.pct_change().dropna()
            sharpe = (rets.mean() / (rets.std() + 1e-6)) * np.sqrt(52)
            dd = (eq_series / eq_series.cummax() - 1.0)
            max_dd = abs(dd.min()) * 100.0

            results.append({
                'Model': m_name.split('(')[0].strip(),
                'Regime': r_name.split('(')[0].strip(),
                'Return (%)': tot_ret,
                'CAGR (%)': cagr,
                'Sharpe': sharpe,
                'Max DD (%)': max_dd
            })

    res_df = pd.DataFrame(results)
    print("\n" + res_df.to_string(index=False))

if __name__ == "__main__":
    analyze_overfitting()
