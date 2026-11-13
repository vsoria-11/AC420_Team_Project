# CAPM Risk Analyzer & Risk-Adjusted Return Recommendation Engine

AC 420 Intro to Financial Analytics. A multi-page **Streamlit** app that measures a stock's market risk with a
single-factor **CAPM regression**, issues a rule-based **Buy / Sell / Hold** signal, and adds a **forward-looking
outlook** (forecast + news context) and an **out-of-sample backtest** to back the signal up.

```
(R_i - R_f) = alpha_i + beta_i (R_M - R_f) + error_i
```

## AI-tag legend (read this first)

Every file and function starts with a comment tag that says who wrote it:

| Tag | Meaning |
|---|---|
| `# [AI-GENERATED]` | Written with AI (Claude); not yet reviewed by the team |
| `# [AI-GENERATED, TEAM-REVIEWED]` | Written with AI; the team has read it and can explain it |
| `# [TEAM-WRITTEN]` | Written by the team |

Each helper file ends with a `[TEAM-WRITTEN]` space for code the team adds. Update tags only to match what really
happened. The "How It Works & AI Use" page contains the full AI-use statement.

## Pages

| Page | What it does |
|---|---|
| **Home** | Ticker + date-range slider; cards for live price, alpha, beta, R², recommendation; scatter plot with OLS line; **Forward-Looking Outlook** (below) |
| **Peer Risk Comparison** | Multi-select stocks; beta and R² bars vs the S&P 500; Security Market Line; Capital Market Line; CSV download |
| **Model Diagnostics** | Full OLS table (coefficients, standard errors, t-stats, p-values); residual plots; plain-English explanations |
| **Backtest** | Walk-forward test of the signal: fit on a trailing window, then measure the next 3 / 6 / 12 months |
| **How It Works & AI Use** | How the recommendation is decided, limitations, AI-use statement |

### Forward-Looking Outlook (Home page)

1. **Statistical signal**: the CAPM Buy/Sell/Hold, unchanged.
2. **Quantitative forward view**: GARCH(1,1) volatility forecast + Monte Carlo simulation (5,000 paths) with the
   CAPM-implied return (Blume-adjusted beta). Shows a fan chart, chance of beating the risk-free rate, chance of
   beating the CAPM return if past alpha continues, 5% Value-at-Risk, analyst target vs CAPM return, and a rolling-beta
   stability note.
3. **News context**: Positive / Negative / Neutral flag from recent yfinance headlines scored with VADER
   (with a few finance words added). Shows "Insufficient news" when fewer than 3 headlines are available.

An **alignment verdict** compares the three layers in plain words (for example BUY + negative news = conflict).
The news flag, forecast and verdict are **context only and never change the Buy/Sell/Hold label**.

## Decision matrix

| Signal | Rule |
|---|---|
| **BUY** | Alpha > 0, statistically significant, and R² >= threshold |
| **SELL** | Alpha < 0 and significant, **or** beta > 1.5 with average excess return <= 0 |
| **HOLD** | Alpha within its standard error of zero, or R² below threshold |

SELL is checked first, then BUY, and everything else is HOLD. The significance level (1%, 5%, 10%) and the R²
threshold are set in the sidebar.

## Project structure

```
Home.py                            # Page 1 (entry point) + Forward-Looking Outlook
pages/
  2_Peer_Risk_Comparison.py
  3_Model_Diagnostics.py
  4_Backtest.py
  5_How_It_Works_and_AI_Use.py
helpers.py                         # sidebar, data download, returns, CAPM, signal, walk-forward backtest
outlook.py                         # GARCH, Monte Carlo, analyst view, news sentiment, alignment verdict
tests/test_app.py                  # unit tests on synthetic data (no internet needed)
requirements.txt
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run Home.py
```

Tests: `pip install -r requirements-dev.txt && pytest`

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to <https://share.streamlit.io> and click **Create app**.
3. Choose the repo and branch, set the main file to `Home.py`, and deploy.

## Data notes

- **Source:** yfinance (adjusted closes). Market = S&P 500 (`^GSPC`); risk-free proxy = 13-week T-bill yield (`^IRX`).
- **WRDS:** the original proposal mentioned WRDS/CRSP/Compustat data (1990-2025). That data is licensed and cannot
  be redistributed in a public repo, so the app uses yfinance. The date slider reaches back to 1990.
- **Excess returns:** stock (or market) return minus the risk-free rate for the same period.
- **CAPM expected return:** `Rf + beta x average market excess return over the window`.
- **Settings across pages:** the sidebar values are kept when you switch pages by re-saving each key in
  `st.session_state` at the top of every page (see `sidebar_settings` in `helpers.py`).

## Limitations

Single factor; index and T-bill proxies; estimates depend on window and frequency; simulation assumes normal shocks;
news is a recent snapshot and cannot be backtested; backtest has small samples, survivorship bias and no trading
costs. Educational project, not investment advice.
