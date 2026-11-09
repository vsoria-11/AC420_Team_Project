"""CAPM regression, recommendation engine and diagnostics (no Streamlit dependency here)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from .data import PERIODS_PER_YEAR


@dataclass
class CAPMResult:
    ticker: str
    freq: str
    ppy: int
    n_obs: int
    alpha: float  # per-period intercept
    beta: float
    r2: float
    adj_r2: float
    alpha_se: float
    beta_se: float
    alpha_t: float
    beta_t: float
    alpha_p: float
    beta_p: float
    beta_vs_1_p: float  # H0: beta = 1
    rf_annual: float
    mrp_annual: float  # market risk premium (annualized mean market excess return)
    expected_return: float  # CAPM: rf + beta * MRP
    realized_return: float  # rf + annualized mean stock excess return
    vol_annual: float
    mkt_return: float
    mkt_vol: float
    data: pd.DataFrame
    model: object  # statsmodels RegressionResultsWrapper

    @property
    def alpha_annual(self) -> float:
        return self.alpha * self.ppy

    @property
    def mean_excess_annual(self) -> float:
        return float(self.data["stock_excess"].mean() * self.ppy)


def run_capm(df: pd.DataFrame, ticker: str, freq: str) -> CAPMResult:
    """OLS:  (R_i - R_f) = alpha + beta * (R_M - R_f) + e."""
    ppy = PERIODS_PER_YEAR[freq]
    X = sm.add_constant(df["market_excess"])
    model = sm.OLS(df["stock_excess"], X).fit()

    alpha, beta = float(model.params["const"]), float(model.params["market_excess"])
    beta_se = float(model.bse["market_excess"])
    beta_vs_1_p = float(2 * stats.t.sf(abs((beta - 1.0) / beta_se), model.df_resid))

    rf_annual = float(df["rf"].mean() * ppy)
    mrp = float(df["market_excess"].mean() * ppy)
    return CAPMResult(
        ticker=ticker,
        freq=freq,
        ppy=ppy,
        n_obs=int(model.nobs),
        alpha=alpha,
        beta=beta,
        r2=float(model.rsquared),
        adj_r2=float(model.rsquared_adj),
        alpha_se=float(model.bse["const"]),
        beta_se=beta_se,
        alpha_t=float(model.tvalues["const"]),
        beta_t=float(model.tvalues["market_excess"]),
        alpha_p=float(model.pvalues["const"]),
        beta_p=float(model.pvalues["market_excess"]),
        beta_vs_1_p=beta_vs_1_p,
        rf_annual=rf_annual,
        mrp_annual=mrp,
        expected_return=rf_annual + beta * mrp,
        realized_return=rf_annual + float(df["stock_excess"].mean() * ppy),
        vol_annual=float(df["stock"].std() * np.sqrt(ppy)),
        mkt_return=rf_annual + mrp,
        mkt_vol=float(df["market"].std() * np.sqrt(ppy)),
        data=df,
        model=model,
    )


# ----------------------------------------------------------------------------
# Recommendation engine
# ----------------------------------------------------------------------------
@dataclass
class Recommendation:
    label: str  # "BUY" | "SELL" | "HOLD"
    reasons: list[str]


def recommend(
    res: CAPMResult, alpha_level: float = 0.10, r2_min: float = 0.30, high_beta: float = 1.5
) -> Recommendation:
    """Decision matrix from the project proposal.

    SELL: significantly negative alpha, OR beta > `high_beta` with zero/negative excess return.
    BUY : significantly positive alpha AND R^2 >= `r2_min`.
    HOLD: everything else (alpha ~ 0 within its standard error, or low R^2).
    """
    sig = res.alpha_p < alpha_level
    reliable = res.r2 >= r2_min
    a_txt = f"{res.alpha_annual:+.2%}/yr"
    p_txt = f"p = {res.alpha_p:.3f}"

    if sig and res.alpha < 0:
        reasons = [f"Alpha is negative ({a_txt}) and statistically significant ({p_txt} < {alpha_level:.2f})."]
        if not reliable:
            reasons.append(f"Caution: R² = {res.r2:.2f} is below {r2_min:.2f}, so the CAPM fit is weak.")
        return Recommendation("SELL", reasons)

    if res.beta > high_beta and res.mean_excess_annual <= 0:
        return Recommendation(
            "SELL",
            [
                f"Beta of {res.beta:.2f} exceeds {high_beta:.1f} (very high systematic risk) while the "
                f"average excess return is {res.mean_excess_annual:+.2%}/yr (zero or negative)."
            ],
        )

    if sig and res.alpha > 0 and reliable:
        return Recommendation(
            "BUY",
            [
                f"Alpha is positive ({a_txt}) and statistically significant ({p_txt} < {alpha_level:.2f}).",
                f"R² = {res.r2:.2f} ≥ {r2_min:.2f}, so market risk explains the stock well enough to trust the estimate.",
            ],
        )

    reasons = []
    if not sig:
        reasons.append(
            f"Alpha ({a_txt}) is not statistically different from zero ({p_txt} ≥ {alpha_level:.2f}); "
            "the stock is priced roughly in line with its CAPM-implied return."
        )
    if sig and res.alpha > 0 and not reliable:
        reasons.append(f"Alpha is positive and significant ({p_txt}), but the model is not reliable enough to act on.")
    if not reliable:
        reasons.append(
            f"R² = {res.r2:.2f} is below {r2_min:.2f}: market movements explain little of this stock's "
            "variance, so beta is statistically uninformative."
        )
    return Recommendation("HOLD", reasons)


def beta_class(beta: float) -> str:
    if beta < 0:
        return "Negative beta"
    if beta > 1.5:
        return "Very high beta"
    if beta > 1.0:
        return "High beta (aggressive)"
    return "Low beta (defensive)"


def r2_class(r2: float) -> str:
    if r2 >= 0.5:
        return "High reliability"
    if r2 >= 0.25:
        return "Moderate reliability"
    return "Low reliability"


def to_row(res: CAPMResult, rec: Recommendation) -> dict:
    return {
        "ticker": res.ticker,
        "beta": res.beta,
        "r2": res.r2,
        "alpha_annual": res.alpha_annual,
        "alpha_p": res.alpha_p,
        "expected_return": res.expected_return,
        "realized_return": res.realized_return,
        "vol_annual": res.vol_annual,
        "rf_annual": res.rf_annual,
        "mrp_annual": res.mrp_annual,
        "mkt_return": res.mkt_return,
        "mkt_vol": res.mkt_vol,
        "n_obs": res.n_obs,
        "recommendation": rec.label,
    }


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def rolling_beta(df: pd.DataFrame, window: int) -> pd.Series:
    cov = df["stock_excess"].rolling(window).cov(df["market_excess"])
    var = df["market_excess"].rolling(window).var()
    return (cov / var).dropna()


def residual_tests(res: CAPMResult) -> pd.DataFrame:
    resid = res.model.resid
    dw = float(durbin_watson(resid))
    jb, jb_p, skew, kurt = jarque_bera(resid)
    bp_lm, bp_p, _, _ = het_breuschpagan(resid, res.model.model.exog)

    dw_note = "No strong autocorrelation" if 1.5 <= dw <= 2.5 else "Possible autocorrelation in residuals"
    return pd.DataFrame(
        [
            {"Test": "Durbin-Watson (autocorrelation)", "Statistic": dw, "p-value": np.nan,
             "Interpretation": dw_note + " (≈2 is ideal)"},
            {"Test": "Jarque-Bera (normality)", "Statistic": float(jb), "p-value": float(jb_p),
             "Interpretation": "Residuals look normal" if jb_p >= 0.05 else
             f"Residuals are non-normal (skew {skew:.2f}, kurtosis {kurt:.2f}); t-tests are approximate"},
            {"Test": "Breusch-Pagan (heteroskedasticity)", "Statistic": float(bp_lm), "p-value": float(bp_p),
             "Interpretation": "Constant variance" if bp_p >= 0.05 else
             "Heteroskedasticity detected; standard errors may be understated"},
        ]
    )
