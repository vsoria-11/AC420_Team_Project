# [AI-GENERATED]  (change to [AI-GENERATED, TEAM-REVIEWED] once your team has reviewed it,
#                  or to [TEAM-WRITTEN] for code you rewrite yourselves)
"""Page 3: the full regression output and what each number means."""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from helpers import DISCLAIMER, download_prices, make_returns, run_capm, sidebar_settings

st.set_page_config(page_title="Model Diagnostics", page_icon="🔬", layout="wide")
settings = sidebar_settings()
ticker = settings["ticker"]

st.title("🔬 Model Diagnostics & Methodology")
if ticker == "":
    st.info("Enter a ticker in the sidebar.")
    st.stop()

try:
    prices = download_prices((ticker,), str(settings["start"]), str(settings["end"]))
    df = make_returns(prices, ticker, settings["freq"])
except ValueError as error:
    st.error(str(error))
    st.stop()

capm = run_capm(df, settings["freq"])
model = capm["model"]

st.subheader(ticker + ": regression results")
st.latex(r"R_i - R_f = \alpha_i + \beta_i (R_M - R_f) + \varepsilon_i")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Beta", f"{capm['beta']:.3f}", help="Slope: how much the stock moves when the market's excess return moves by 1.")
c2.metric("Alpha (per period)", f"{capm['alpha']:+.4%}", help="Intercept: average return the market does not explain.")
c3.metric("R²", f"{capm['r2']:.3f}", f"adjusted {capm['adj_r2']:.3f}", delta_color="off",
          help="Share of the stock's variance explained by the market. The rest is company-specific risk.")
c4.metric("Observations", capm["n_obs"])

# ---- Coefficient table: coefficient, standard error, t-stat, p-value ----------------------------
st.markdown("#### Coefficient table")
interval = model.conf_int()
table = pd.DataFrame({
    "Coefficient": model.params.values,
    "Std. error": model.bse.values,
    "t-statistic": model.tvalues.values,
    "p-value": model.pvalues.values,
    "95% CI low": interval[0].values,
    "95% CI high": interval[1].values,
}, index=["Alpha (intercept)", "Beta (market excess return)"])
st.dataframe(table.style.format("{:.4f}"))

with st.expander("Full statsmodels OLS summary"):
    st.code(model.summary().as_text())

# ---- Residual plots: do the errors look random? -------------------------------------------------
st.markdown("#### Residuals (the part of each return CAPM does not explain)")
col1, col2 = st.columns(2)
with col1:
    fig = go.Figure(go.Scatter(x=model.fittedvalues * 100, y=model.resid * 100, mode="markers"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Residuals vs fitted values", xaxis_title="Fitted (%)", yaxis_title="Residual (%)")
    st.plotly_chart(fig)
with col2:
    fig2, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(model.resid * 100, bins=20, edgecolor="white")
    ax.set_title("Histogram of residuals")
    ax.set_xlabel("Residual (%)")
    st.pyplot(fig2)
    plt.close(fig2)
st.caption("Healthy residuals scatter randomly around zero with no pattern and look roughly bell-shaped.")

# ---- Educational explanations -------------------------------------------------------------------
st.divider()
st.subheader("How to read these numbers")
with st.expander("Beta: market (systematic) risk", expanded=True):
    st.markdown("Beta is the slope of the regression. **Above 1**: the stock amplifies market moves (aggressive). "
                "**Below 1**: it dampens them (defensive). Beta measures only the risk that cannot be diversified away.")
with st.expander("R²: how much can we trust beta?"):
    st.markdown("R² is the share of the stock's movement explained by the market. High R² means the one-factor model "
                "describes the stock well. Low R² means most movement is company-specific, so beta is not very informative.")
with st.expander("Alpha: return above what risk predicts"):
    st.markdown("Alpha is the intercept. It only matters if it is **statistically significant**: compare its p-value "
                "with the significance level in the sidebar. Over short windows alpha is noisy, so most stocks show "
                "an insignificant alpha, which leads to HOLD.")
with st.expander("Limits of the model"):
    st.markdown("- One factor only: CAPM ignores size, value and momentum.\n"
                "- The S&P 500 and the 13-week T-bill are proxies for the market and the risk-free rate.\n"
                "- Results change with the date range and return frequency.\n"
                "- Past alpha does not guarantee future returns.")

st.caption(DISCLAIMER)
