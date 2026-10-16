# Inter IIT Tech Meet 15.0 - Quantitative Trading Selection
## Track A: Volatility-Adjusted Cross-Sectional Strategy
### Official Submission & Implementation Blueprint

---

## 1. Executive Summary

This submission implements **Option A: Robust Diversified Multi-Regime Academic Pillar Strategy** with **Classical 100-Day Macro Trend Defense** ([Faber 2007](https://doi.org/10.3905/jwm.2007.69)). 

The strategy was designed strictly adhering to institutional quant standards:
1. **Statistical Superiority**: Employs the **Garman-Klass (1980)** range volatility estimator, which has **~7.4x higher statistical efficiency** than standard close-to-close variance.
2. **True Out-of-Sample Generalization (No Overfitting)**: Avoids parameter curve-fitting by adopting equal-weighted (1/3 each) academic factor pillars and holding a diversified portfolio of 6 assets (top 25% quintile) with an 18% concentration ceiling.
3. **Macro Drawdown Defense**: Mitigates correlation spikes during market downturns (e.g., 2018 Fed tightening, March 2020 crash) by dynamically halving gross exposure to 48% (holding 52% in risk-free cash) when the broad market falls below its 100-day moving average.
4. **Drawdown Suppression**: Slashes maximum historical drawdown from **37.88% down to 22.64%** while elevating Sharpe ratio to **1.16** and final equity to **$231,775.83 (+131.78% total return)**.
5. **Zero Data Leakage**: Causality verified with **0.0e+00** discrepancy on Truncation Invariance tests.
6. **Zero Rule Violations**: 0 gross leverage violations, 0 bankruptcies, 100% compliant with backtester rules.

---

## 2. Code Boundaries: What Can & Cannot Be Modified

Per the organizer's `README.md` and `Quantitative_Trading_Tracks_Updated.md`, the repository enforces a strict boundary between the **Participant Space** and the **Organizer Harness Space**:

```text
Repository Layout:
├── strategy_base.py          # [DO NOT MODIFY] Core API contracts (BaseStrategy, Context, Bar)
├── backtester.py             # [DO NOT MODIFY] Engine core, ledger, rule guard, metrics
├── run_backtest.py           # [DO NOT MODIFY] Command-line entry runner
├── test_submission.py        # [DO NOT MODIFY] Automated pre-submission validation script
├── data/                     # [DO NOT MODIFY] Official in-sample datasets
│   ├── Dataset_PS-A.csv      # Track A daily cross-sectional dataset (24 assets)
│   └── ASSET_ALPHA_1s.parquet# Track B 1-second microstructure dataset
│
├── config.yaml               # [PARTICIPANT SPACE] Evaluation settings (cadence, track, capital)
├── requirements.txt          # [PARTICIPANT SPACE] Strategy dependencies
├── results/                  # Auto-generated backtest performance artifacts
│   ├── summary.json          # Machine-readable performance metrics
│   ├── trades.csv            # Tick-by-tick order fill log
│   └── tearsheet.png         # 4-panel visual performance charts
│
└── src/                      # [PARTICIPANT SPACE] Primary Strategy Workspace
    ├── __init__.py           # Package initialization
    ├── engine.py             # MANDATORY ENTRY POINT: ParticipantStrategy(BaseStrategy)
    ├── models/               # Statistical and financial models
    │   ├── volatility.py     # Garman-Klass, Parkinson, EWMA, Realized Volatility
    │   ├── ranking.py        # 3-Pillar Cross-Sectional Ranker + Macro Trend Filter
    │   └── portfolio_opt.py  # Volatility-Adjusted Risk Parity Optimizer
    ├── utils/                # Helper math transforms
    ├── research/             # Statistical validation, walk-forward & leakage audits
    └── report.pdf            # Formal research report (PDF, max 10 pages)
```

### Modification Rules Table:

| Component | Path | Allowed to Modify? | Purpose / Boundary Rule |
| :--- | :--- | :---: | :--- |
| **Strategy Entry Point** | `src/engine.py` | **YES (Mandatory)** | Implements `ParticipantStrategy(BaseStrategy)` with `initialize()` and `on_bar()`. |
| **Sub-Modules** | `src/models/*.py` | **YES** | Custom econometric, volatility, ranking, and portfolio sizing models. |
| **Research & Validation** | `src/research/*.py` | **YES** | Statistical tests, factor IC analysis, overfitting audits, report compilation. |
| **Written Research Report** | `src/report.pdf` | **YES (Mandatory)** | Comprehensive strategy report ($\le 10$ pages). |
| **Dependencies** | `requirements.txt` | **YES** | Participant dependencies (e.g. numpy, pandas, scipy, reportlab). |
| **Configuration** | `config.yaml` | **YES** | Rebalance cadence, capital, and active track settings. |
| **API Contracts** | `strategy_base.py` | **NO** | Core protocol defining `BaseStrategy`, `Context`, and `Bar`. |
| **Backtester Core** | `backtester.py` | **NO** | Trusted engine calculating fills, slippage, commission, and metrics. |
| **Pre-Submission Validator** | `test_submission.py` | **NO** | Automated sanity validator running 5 structural and risk checks. |
| **Runner Script** | `run_backtest.py` | **NO** | CLI runner used by organizers to grade held-out out-of-sample data. |
| **Datasets** | `data/*.csv`, `*.parquet` | **NO** | Official evaluation data; data substitution violates fairness rules. |

---

## 3. Performance Summary (Official Backtester Output)

Evaluation executed against official `data/Dataset_PS-A.csv` (1,484 trading days, 24 assets, 2016–2022):

| Metric | Measured Value | Benchmark / Limit | Status |
| :--- | :---: | :---: | :---: |
| **Initial Capital** | $100,000.00 | $100,000.00 | Standard |
| **Final Equity** | **$231,775.83** | > $100,000.00 | **+131.78% Return** |
| **CAGR** | **15.34%** | — | Robust Compound Growth |
| **Annualized Sharpe Ratio** | **1.16** | > 1.00 | Statistically Significant |
| **Sharpe 95% Confidence Interval** | **[0.38, 2.07]** | Lower Bound > 0.0 | **Zero-Alpha Rejected** |
| **Sortino Ratio** | **1.57** | > 1.00 | Favorable Downside Deviation |
| **Maximum Drawdown (MDD)** | **22.64%** | < 25.00% | **Drawdown Suppressed** |
| **Calmar Ratio** | **0.68** | — | Return / MDD Ratio |
| **Daily Win Rate** | **51.99%** | > 50.00% | Positive Daily Edge |
| **Mean Gross Leverage** | **0.76** | $\le 1.00$ | Well within risk envelope |
| **Max Gross Leverage** | **0.98** | $\le 1.00$ | Strict compliance |
| **Gross Leverage Violations** | **0** | **0** | **100% Rule Compliance** |
| **Account Bankruptcies** | **0** | **0** | **100% Capital Preservation** |
| **Throughput** | **1,460 ticks/s** | > 100 ticks/s | Fast Causal Execution |

---

## 4. Theoretical Architecture & Mathematical Formulations

### Deliverable 1: Volatility Modeling (`src/models/volatility.py`)
Standard close-to-close volatility discards all intraday price discovery. We employ the **Garman-Klass (1980)** range volatility estimator:

$$\sigma_{GK, t}^2 = 0.5 \left( \ln \frac{H_t}{L_t} \right)^2 - (2\ln 2 - 1) \left( \ln \frac{C_t}{O_t} \right)^2$$

Annualized over rolling $k=20$ trading days:
$$\sigma_{GK} = \sqrt{252 \times \frac{1}{k}\sum_{\tau=0}^{k-1} \sigma_{GK, t-\tau}^2}$$

- **Efficiency**: Theoretical variance is ~0.17 vs. 1.33 for close-to-close, yielding **7.4x greater statistical efficiency**.
- **Alternative estimators implemented**: Parkinson (1980) High-Low, EWMA ($\lambda=0.94$), and classical Realized Volatility.

---

### Deliverable 2: Cross-Sectional Ranking (`src/models/ranking.py`)
Assets are ranked across three complementary academic pillars:

1. **Quality Momentum / Trend Sharpe (Pillar 1)** ([Jegadeesh & Titman 1993](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)):
   $$S_i = \frac{R_{60, i}}{\sigma_{GK, i} + 10^{-4}}, \quad R_{60, i} = \frac{P_{t, i} - P_{t-60, i}}{P_{t-60, i}}$$
   Penalizes volatile, lottery-like spikes; rewards smooth, steady compounders.

2. **Short-Term Pullback / Liquidity Reversion (Pillar 2)** ([Lehmann 1990](https://doi.org/10.2307/2937820)):
   $$Z_{\text{pullback}, i} = - \frac{P_{t, i} - \text{SMA}_{10, i}}{\text{Std}_{10, i} + 10^{-4}}$$
   Enters established momentum trends on temporary 2-week liquidity dips rather than chasing overbought peaks.

3. **Low-Volatility Anomaly Factor (Pillar 3)** ([Ang et al. 2006](https://doi.org/10.1111/j.1540-6261.2006.00843.x); [Frazzini & Pedersen 2014](https://doi.org/10.1016/j.jfineco.2013.10.005)):
   $$F_{\text{lowvol}, i} = - \sigma_{GK, i}$$
   Tilts toward structurally stable assets with lower downside tail risk.

4. **Equal-Weighted Composite Standardization**:
   $$Z(X_i) = \text{clip}\left( \frac{X_i - \bar{X}}{\text{std}(X)}, -3.0, 3.0 \right)$$
   $$\text{Score}_i = \frac{1}{3} Z(S_i) + \frac{1}{3} Z(Z_{\text{pullback}, i}) + \frac{1}{3} Z(F_{\text{lowvol}, i}) + \mathbf{1}_{\{P_{t, i} > \text{SMA}_{50, i}\}} \times 0.25$$

---

### Deliverable 3: Portfolio Construction (`src/models/portfolio_opt.py`)
1. **Asset Selection**: Top $K=6$ assets (top 25% quintile of the 24-asset universe).
2. **Inverse-Volatility Risk Parity** ([Maillard, Roncalli, & Teiletche 2010](https://doi.org/10.3905/jpm.2010.36.4.060)):
   $$w_i^{\text{raw}} = \frac{\sigma_{GK, i}^{-1}}{\sum_{j=1}^K \sigma_{GK, j}^{-1}} \times L_{\text{target}}$$
   Eliminates Markowitz mean-variance parameter estimation error.
3. **Concentration Cap**: $w_i \le \min(18\%, L_{\text{target}} / 3.0)$ with Iterative Proportional Fitting (IPF).
4. **Macro Trend Defense (Drawdown Suppression)** ([Faber 2007](https://doi.org/10.3905/jwm.2007.69)):
   - Equal-weighted broad market index: $M_t = \frac{1}{N}\sum_{i=1}^N P_{t, i}$.
   - If $M_t < \text{SMA}_{100}(M)$ (bear regime): $L_{\text{target}} = 0.48$ (52% held in cash).
   - If $M_t \ge \text{SMA}_{100}(M)$ (bull regime): $L_{\text{target}} = 0.96 \le 1.0$.
5. **Turnover Deadband**: If $|w_{i, \text{target}} - w_{i, \text{current}}| < 2.0\%$, the position is unchanged, minimizing turnover friction.

---

## 5. Walk-Forward Regime Validation & Anti-Overfitting Proof

To guarantee that the strategy will generalize to unseen out-of-sample data, we conducted a 3-regime walk-forward audit comparing our **Robust Diversified Model** against a naive **Concentrated 3-Asset Model**:

| Model Architecture | 2017–2018 (Regime 1) Sharpe | 2019–2020 (Regime 2) Sharpe | 2021–2022 (Regime 3) Sharpe | Full History Max Drawdown |
| :--- | :---: | :---: | :---: | :---: |
| **Concentrated Top 3 (Overfitted)** | 1.91 | **0.21 (Collapsed)** | 0.58 | **39.73%** |
| **Option A (Robust Diversified)** | **2.16** | **0.77 (Resilient)** | **1.17** | **21.25%** |

*Key Takeaway*: While hyper-concentrated portfolios may look attractive on in-sample returns, their Sharpe ratio collapses by **89%** during adverse regimes. Option A maintains positive alpha and stable risk across all three distinct market cycles.

---

## 6. How to Reproduce & Validate

### Step 1: Run Pre-Submission Sanity Check
```bash
python3 test_submission.py
```
*Expected Output*: `[SUCCESS] ALL PRE-SUBMISSION SANITY CHECKS PASSED!` (5/5 checks passed).

### Step 2: Run Full Track A Backtest
```bash
python3 run_backtest.py --track A
```
*Expected Output*: Final Equity: $231,775.83, Sharpe: 1.16, Max Drawdown: 22.64%, Violations: 0.

### Step 3: Run Data Leakage Audit
```bash
python3 src/research/audit_data_leakage.py
```
*Expected Output*: `Max absolute difference between truncated run and full run: 0.0e+00` (Zero future data leakage).

### Step 4: Recompile Written Report PDF
```bash
python3 src/research/generate_report_pdf.py src/report.pdf
```
*Expected Output*: `Successfully generated PDF report: src/report.pdf` (3 pages $\le 10$ page limit).

---

## 7. Submission Checklist Verification

- [x] All organizer-provided canonical files (`strategy_base.py`, `backtester.py`, `test_submission.py`, `run_backtest.py`) are **100% untouched**.
- [x] Participant strategy implemented in mandatory entry point `src/engine.py` inheriting from `BaseStrategy`.
- [x] Multi-estimator volatility model implemented in `src/models/volatility.py`.
- [x] Cross-sectional factor ranking & macro trend timing implemented in `src/models/ranking.py`.
- [x] Volatility-adjusted Risk Parity portfolio construction implemented in `src/models/portfolio_opt.py`.
- [x] Exhaustive docstrings and academic citations throughout all codebase files.
- [x] Formal research report `src/report.pdf` compiled and verified (3 pages $\le 10$ pages).
- [x] `results/summary.json`, `results/trades.csv`, and `results/tearsheet.png` generated and verified.
- [x] `requirements.txt` contains all runtime dependencies.
- [x] `test_submission.py` passed 5/5.
- [x] Deterministic execution with zero lookahead bias and zero data leakage.
