# CAPM Risk Analyzer & Risk-Adjusted Return Recommendation Engine

AC 420 Intro to Financial Analytics: an interactive multi-page **Streamlit** app that estimates a stock's
systematic risk with a single-factor **Capital Asset Pricing Model (CAPM)** regression and issues rule-based
**Buy / Sell / Hold** signals from risk-adjusted abnormal return (alpha).

```
(R_i - R_f) = α_i + β_i (R_M - R_f) + ε_i
```

## Features

| Page | What it does |
|---|---|
| **Home: Single Stock Analyzer** | Ticker input and date-range slider; metric cards for live price, α, β, R², and the recommendation; excess-return scatterplot with the OLS line; CAPM expected-return forecast |
| **Peer Risk Comparison** | Multi-select tickers; side-by-side β and R² bars vs. the S&P 500; Security Market Line (β vs. return); Capital Market Line (volatility vs. return); CSV export |
| **Model Diagnostics** | Full OLS table (coefficients, std. errors, t-stats, p-values, CIs); Durbin-Watson, Jarque-Bera and Breusch-Pagan tests; residual plots; rolling beta; methodology and limitations |

## Decision matrix

| Signal | Rule |
|---|---|
| **BUY** | α > 0, statistically significant, and R² ≥ threshold |
| **SELL** | α < 0 and significant, **or** β > 1.5 with average excess return ≤ 0 |
| **HOLD** | α within its standard error of zero, or R² below threshold (market risk does not reliably explain the stock) |

Significance level (1%, 5%, or 10%) and the R² threshold are adjustable in the sidebar.
SELL conditions are checked first, then BUY, and everything else is HOLD.

## Project structure

```
Home.py                         # Page 1 (Streamlit entry point)
pages/
  2_Peer_Risk_Comparison.py     # Page 2
  3_Model_Diagnostics.py        # Page 3
utils/
  data.py                       # yfinance download, returns & excess-return construction
  capm.py                       # OLS regression, recommendation engine, diagnostics
  charts.py                     # Plotly charts
  ui.py                         # Shared sidebar settings
tests/test_capm.py              # Unit tests on synthetic data
requirements.txt
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run Home.py
```

Run the tests with `pip install -r requirements-dev.txt && pytest`.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to <https://share.streamlit.io> and click **Create app**.
3. Pick the repo and branch, set the main file path to `Home.py`, and deploy.

## Data & method notes

- **Market data:** `yfinance` (adjusted closes). Market = S&P 500 (`^GSPC`); risk-free proxy = 13-week T-bill yield (`^IRX`).
- **Returns:** simple returns at daily, weekly, or monthly frequency; excess returns subtract the risk-free rate
  (annualized yield ÷ periods per year, lagged one period).
- **CAPM expected return:** `Rf + β × (average market excess return over the window)`.
- **Alpha** is reported annualized (per-period intercept × periods per year).
- **Historical baseline (1990–2025, WRDS):** WRDS/CRSP/Compustat data is licensed and cannot be redistributed on a
  public repo, so the app pulls its estimation window from yfinance. The date slider reaches back to 1990.

## Limitations

Single-factor model; index and T-bill proxies; estimates depend on window and frequency; OLS assumptions may not hold
(see diagnostics page); past alpha does not predict future returns. Educational use only, not investment advice.
