# [AI-GENERATED]  (change to [AI-GENERATED, TEAM-REVIEWED] once your team has reviewed it,
#                  or to [TEAM-WRITTEN] for code you rewrite yourselves)
"""Page 5: how the recommendation is decided + AI-use statement (requested by the professor)."""

import streamlit as st

from helpers import DISCLAIMER

st.set_page_config(page_title="How It Works & AI Use", page_icon="📘", layout="wide")

# ==========================================================================================
# EDIT THIS BLOCK so the AI-use statement matches what your team really did.
# Anything still in [BRACKETS] shows on the page as a reminder.
# ==========================================================================================
AI_TOOL = "Claude (Anthropic)"
AI_GENERATED = [
    "The Python and Streamlit code for the app: data download, CAPM regression, decision logic, charts, "
    "forecast, news scoring, backtest, and pages",
    "The unit tests and the README",
    "A first draft of the text on this page",
    "[EDIT: add or remove items so this list is accurate]",
]
TEAM_CONTRIBUTIONS = [
    "The project proposal and specifications",
    "Choosing the model (single-factor CAPM regression) and the data sources",
    "Defining the Buy / Sell / Hold decision rules and thresholds",
    "Directing the work, testing the app and checking the results",
    "The decision to add the news-sentiment layer as context only",
    "[EDIT: add anything else your team did, e.g. code you wrote or rewrote yourselves]",
]
CLASS_NOTE = ("[CONFIRM BEFORE SUBMITTING: Our team covered the basic topics (regression, returns, "
              "Streamlit) in class, and we reviewed the code and can explain it.]")
# ==========================================================================================

st.title("📘 How It Works & AI Use")

tab1, tab2, tab3 = st.tabs(["How the recommendation is decided", "Limitations", "AI-use statement"])

with tab1:
    st.header("From data to a recommendation")
    st.markdown("""
**1. Data.** Daily prices for the stock, the S&P 500 (^GSPC) and the 13-week T-bill yield (^IRX) come from Yahoo
Finance (yfinance). Prices become returns, and subtracting the risk-free rate gives **excess returns**.

**2. CAPM regression.** We fit a straight line through the stock's excess returns versus the market's excess returns
(ordinary least squares, OLS):
""")
    st.latex(r"R_i - R_f = \alpha_i + \beta_i (R_M - R_f) + \varepsilon_i")
    st.markdown("""
- **Beta** is the slope: how strongly the stock moves with the market (market risk).
- **Alpha** is the intercept: return above or below what the stock's risk predicts.
- **R²** is how much of the stock's movement the market explains (how far to trust beta).
- **p-values** tell us whether alpha is statistically different from zero or just noise.

**3. Buy / Sell / Hold decision matrix (numbers only).**

| Signal | Rule |
|---|---|
| **BUY** | Alpha above zero, statistically significant, and R² at or above the threshold |
| **SELL** | Alpha below zero and significant, **or** beta above 1.5 with zero or negative average excess return |
| **HOLD** | Alpha within its standard error of zero, or R² below the threshold |

The significance level and R² threshold are set in the sidebar. **Only this step decides the label.**

**4. Forward-looking outlook (context, never changes the label).**
- *Forecast volatility* with a GARCH(1,1) model, which lets volatility rise and fall over time.
- *Monte Carlo simulation* of 5,000 possible price paths using the CAPM-implied return (with a Blume-adjusted beta,
  0.67 x raw + 0.33 x 1). From it we read the chance of beating the risk-free rate, the 5% Value-at-Risk and a fan chart.
- *Second scenario:* the same simulation with the historical alpha added, to ask "what if past alpha continues?"
- *Analyst price target* from Yahoo Finance compared with the CAPM-required return, when available.
- *Forecast view:* Favorable if the chance of beating the risk-free rate is 60% or more, Unfavorable if under 50%,
  otherwise Mixed.

**5. News layer (context only).** Recent headlines from yfinance are scored with VADER (a sentiment dictionary, extended
with some finance words). The average becomes Positive, Neutral or Negative. With fewer than 3 headlines we show
"Insufficient news". It is a snapshot of recent headlines, not a forecast.

**6. Alignment verdict (advisory).** A short sentence compares the signal, the forecast and the news. For example,
BUY with negative news is shown as a **conflict** to investigate before buying; BUY with a favorable forecast and
positive news is **strong alignment**. The verdict never changes the Buy/Sell/Hold label.

**7. Backtest.** The Backtest page tests the CAPM signal on data the model has not seen, by fitting on a trailing
window and checking the next 3, 6 or 12 months.
""")

with tab2:
    st.header("Limitations")
    st.markdown("""
- **Single factor.** CAPM ignores other known drivers of returns such as size, value and momentum.
- **Proxies.** The S&P 500 stands in for the market portfolio, and the 13-week T-bill yield stands in for the risk-free rate.
- **Window dependence.** Alpha and beta change with the date range and the return frequency.
- **Noisy alpha.** Over short windows most alphas are not statistically significant, so many stocks get HOLD.
- **Forecast assumptions.** The simulation assumes normally distributed shocks and a constant expected return, which is a
  simplification. The "past alpha continues" scenario assumes something the backtest may not support.
- **News is a snapshot.** Headline sentiment is approximate, covers only recent headlines and cannot be backtested.
- **Backtest caveats.** Results may be mixed, samples are small, stocks are chosen from today's survivors, and trading
  costs are ignored.
- **Data source.** The original proposal mentioned WRDS/CRSP data (1990-2025). That data is licensed and cannot be
  redistributed on a public repository, so the app uses yfinance.
- **Not investment advice.** This is an educational project.
""")

with tab3:
    st.header("AI-use statement")
    st.markdown("**AI tool used:** " + AI_TOOL)
    st.markdown("**What the AI generated:**")
    for item in AI_GENERATED:
        st.markdown("- " + item)
    st.markdown("**What our team contributed:**")
    for item in TEAM_CONTRIBUTIONS:
        st.markdown("- " + item)
    st.markdown("**Understanding of the code:** " + CLASS_NOTE)
    st.markdown("**Code tags in the GitHub repository:** every file and function starts with a comment tag. "
                "`# [AI-GENERATED]` = written by AI, not yet reviewed by the team. "
                "`# [AI-GENERATED, TEAM-REVIEWED]` = written by AI and reviewed by the team. "
                "`# [TEAM-WRITTEN]` = written by the team.")

st.caption(DISCLAIMER)
