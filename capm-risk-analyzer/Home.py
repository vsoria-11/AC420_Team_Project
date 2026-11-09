"""Page 1: Home & single-stock CAPM analyzer."""
import streamlit as st

from utils.capm import beta_class, r2_class, recommend, run_capm
from utils.charts import scatter_capm
from utils.data import DataError, get_live_quote, get_returns
from utils.ui import DISCLAIMER, banner, sidebar_settings

st.set_page_config(page_title="CAPM Risk Analyzer", page_icon="📈", layout="wide")
s = sidebar_settings()

st.title("📈 CAPM Risk Analyzer & Risk-Adjusted Return Recommendation Engine")
st.markdown(
    "Estimate a stock's **systematic risk (β)**, **risk-adjusted abnormal return (α)** and **model fit (R²)** "
    "with a single-factor CAPM regression, then get a rule-based **Buy / Sell / Hold** signal. "
    "Use the sidebar to change the ticker, estimation window, and decision thresholds."
)

if not s.ticker:
    st.info("Enter a ticker symbol in the sidebar to begin.")
    st.stop()

try:
    with st.spinner(f"Downloading data and fitting CAPM for {s.ticker}..."):
        df = get_returns(s.ticker, s.start, s.end, s.freq)
        res = run_capm(df, s.ticker, s.freq)
except DataError as exc:
    st.error(str(exc))
    st.stop()

rec = recommend(res, s.alpha_level, s.r2_min)

st.subheader(f"{s.ticker} — CAPM summary")
c1, c2, c3, c4, c5 = st.columns(5)

quote = get_live_quote(s.ticker)
if quote:
    last, prev = quote
    c1.metric("Live Price", f"${last:,.2f}", f"{last / prev - 1:+.2%} today",
              help="Latest price from Yahoo Finance (may be delayed ~15 min).")
else:
    c1.metric("Live Price", "N/A", help="Live quote unavailable from Yahoo Finance right now.")

c2.metric("Alpha (α, annualized)", f"{res.alpha_annual:+.2%}", f"p = {res.alpha_p:.3f}", delta_color="off",
          help="Return earned above/below what CAPM predicts for the stock's risk. "
               f"The per-period ({s.freq.lower()}) alpha is {res.alpha:+.3%}.")
c3.metric("Beta (β)", f"{res.beta:.2f}", beta_class(res.beta), delta_color="off",
          help="Sensitivity to S&P 500 excess returns. β > 1 amplifies market moves; β < 1 dampens them.")
c4.metric("R²", f"{res.r2:.2f}", r2_class(res.r2), delta_color="off",
          help="Share of the stock's return variance explained by the market. Low R² = beta is unreliable.")
c5.metric("Recommendation", rec.label,
          help="Rule-based signal using alpha significance, R², and beta. See Model Diagnostics for the rules.")

banner(rec.label, f"**{rec.label}** — " + " ".join(rec.reasons))

left, right = st.columns([3, 2])
with left:
    st.plotly_chart(scatter_capm(res))
with right:
    st.markdown("#### Return forecast (CAPM)")
    st.metric("CAPM expected return (annual)", f"{res.expected_return:.2%}",
              help="E[R] = Rf + β × (average market excess return over the window).")
    st.metric("Realized return (annual, avg.)", f"{res.realized_return:.2%}", f"{res.alpha_annual:+.2%} vs expected",
              delta_color="off")
    st.markdown(
        f"""
| Input | Value |
|---|---|
| Risk-free rate (avg. ^IRX) | {res.rf_annual:.2%} |
| Market risk premium (S&P 500) | {res.mrp_annual:.2%} |
| β 95% interval | {res.beta - 1.96 * res.beta_se:.2f} to {res.beta + 1.96 * res.beta_se:.2f} |
| Observations | {res.n_obs} {s.freq.lower()} |
| Window | {df.index[0].date()} → {df.index[-1].date()} |
"""
    )

with st.expander("How is the recommendation decided?"):
    st.markdown(
        f"""
- **BUY**: α > 0, statistically significant (p < {s.alpha_level:.2f}), and R² ≥ {s.r2_min:.2f}.
- **SELL**: α < 0 and significant, **or** β > 1.5 with zero/negative average excess return.
- **HOLD**: α is within its standard error of zero, or R² is too low for beta to be trusted.
"""
    )

st.caption(DISCLAIMER)
