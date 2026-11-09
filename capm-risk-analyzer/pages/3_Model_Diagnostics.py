"""Page 3: Model diagnostics & methodology."""
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sps
import streamlit as st

from utils.capm import recommend, residual_tests, rolling_beta, run_capm
from utils.charts import residuals_vs_fitted, rolling_beta_chart
from utils.data import DataError, get_returns
from utils.ui import DISCLAIMER, sidebar_settings

st.set_page_config(page_title="Model Diagnostics", page_icon="🔬", layout="wide")
s = sidebar_settings()

st.title("🔬 Model Diagnostics & Methodology")

if not s.ticker:
    st.info("Enter a ticker symbol in the sidebar.")
    st.stop()

try:
    df = get_returns(s.ticker, s.start, s.end, s.freq)
    res = run_capm(df, s.ticker, s.freq)
except DataError as exc:
    st.error(str(exc))
    st.stop()

st.subheader(f"{s.ticker}: OLS regression results")
st.latex(r"R_i - R_f = \alpha_i + \beta_i\,(R_M - R_f) + \varepsilon_i")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Beta", f"{res.beta:.3f}", help="Slope of the regression: sensitivity to market excess returns.")
m2.metric("Alpha (per period)", f"{res.alpha:+.4%}", help="Intercept: average return unexplained by market risk.")
m3.metric("R²", f"{res.r2:.3f}", f"adj. {res.adj_r2:.3f}", delta_color="off",
          help="Fraction of return variance explained by the market (the rest is idiosyncratic risk).")
m4.metric("Observations", res.n_obs)

st.markdown("#### Coefficient table")
ci = res.model.conf_int()
coef = pd.DataFrame(
    {
        "Coefficient": res.model.params.values,
        "Std. Error": res.model.bse.values,
        "t-statistic": res.model.tvalues.values,
        "p-value": res.model.pvalues.values,
        "95% CI low": ci[0].values,
        "95% CI high": ci[1].values,
    },
    index=["Alpha (intercept)", "Beta (market excess return)"],
)
st.dataframe(coef.style.format("{:.4f}"))
st.caption(
    f"Test of H₀: β = 1 → p-value {res.beta_vs_1_p:.3f} "
    + ("(beta is significantly different from the market)." if res.beta_vs_1_p < 0.05
       else "(cannot reject that this stock moves one-for-one with the market).")
)

with st.expander("Full statsmodels OLS summary"):
    st.code(res.model.summary().as_text())

st.markdown("#### Residual diagnostics")
st.dataframe(residual_tests(res).style.format({"Statistic": "{:.3f}", "p-value": "{:.3f}"}, na_rep="—"),
             hide_index=True)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(residuals_vs_fitted(res))
with c2:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    axes[0].hist(res.model.resid * 100, bins=20, color="#3b82f6", edgecolor="white")
    axes[0].set(title="Residual histogram", xlabel="Residual (%)")
    sps.probplot(res.model.resid, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q plot")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("#### Beta stability")
max_w = len(df) // 2
if max_w >= 12:
    default_w = {"Daily": 126, "Weekly": 52, "Monthly": 36}[s.freq]
    win = st.slider("Rolling window (periods)", 12, max_w, min(default_w, max_w))
    st.plotly_chart(rolling_beta_chart(rolling_beta(df, win), res.beta, s.ticker))
else:
    st.info("Not enough observations for a rolling-beta chart. Widen the date range.")

st.divider()
st.subheader("Methodology & interpretation guide")

rec = recommend(res, s.alpha_level, s.r2_min)
st.markdown(f"**Current signal for {s.ticker}: {rec.label}** — " + " ".join(rec.reasons))

with st.expander("Beta (β): systematic risk", expanded=True):
    st.markdown(
        """
Beta is the slope of the regression of the stock's excess returns on the market's excess returns.
**β > 1** means the stock tends to amplify market moves (aggressive); **β < 1** means it dampens them (defensive);
**β ≈ 0** means little market sensitivity; **β < 0** means it tends to move against the market.
Beta only measures *systematic* risk, the part that cannot be diversified away.
"""
    )
with st.expander("R²: how much can we trust the beta?"):
    st.markdown(
        """
R² is the share of the stock's return variance explained by market movements. A high R² means the single-factor
model describes the stock well. A low R² means most movement is company-specific, so beta (and the CAPM-implied
return built on it) is statistically uninformative. The app downgrades signals when R² is below your threshold.
"""
    )
with st.expander("Alpha (α): risk-adjusted abnormal return"):
    st.markdown(
        """
Alpha is the regression intercept: the average return above (or below) what CAPM predicts given the stock's beta.
It is only meaningful if it is **statistically significant**: compare the p-value to your chosen significance level.
Over short windows alpha is noisy, so most stocks will show an insignificant alpha, which maps to **HOLD**.
"""
    )
with st.expander("Decision matrix"):
    st.markdown(
        f"""
| Signal | Rule |
|---|---|
| **BUY** | α > 0, p < {s.alpha_level:.2f}, and R² ≥ {s.r2_min:.2f} |
| **SELL** | α < 0 and p < {s.alpha_level:.2f}, **or** β > 1.5 with average excess return ≤ 0 |
| **HOLD** | α within its standard error of zero, or R² below {s.r2_min:.2f} |
"""
    )
with st.expander("Data & model limitations"):
    st.markdown(
        """
- **Single factor.** CAPM ignores size, value, momentum, and other priced factors (see Fama-French).
- **Proxies.** The S&P 500 stands in for the market portfolio and the 13-week T-bill yield (^IRX) for the risk-free rate.
- **Estimation window.** Beta and alpha change with the sample period and return frequency; check the rolling-beta chart.
- **Statistical assumptions.** OLS standard errors assume homoskedastic, uncorrelated residuals; see the tests above.
- **Backward-looking.** Historical alpha does not guarantee future performance.
"""
    )

st.caption(DISCLAIMER)
