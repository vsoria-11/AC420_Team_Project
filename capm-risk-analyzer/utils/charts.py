"""Plotly chart builders."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .capm import CAPMResult

REC_COLORS = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#ca8a04"}


def scatter_capm(res: CAPMResult) -> go.Figure:
    df = res.data
    x, y = df["market_excess"] * 100, df["stock_excess"] * 100
    xs = np.linspace(x.min(), x.max(), 50)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers", name="Observations",
            marker=dict(size=7, opacity=0.6, color="#2563eb"),
            customdata=df.index.strftime("%Y-%m-%d"),
            hovertemplate="%{customdata}<br>Market: %{x:.2f}%<br>Stock: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs, y=res.alpha * 100 + res.beta * xs, mode="lines",
            name=f"OLS fit: α={res.alpha * 100:.2f}%, β={res.beta:.2f}",
            line=dict(color="#dc2626", width=3),
        )
    )
    fig.update_layout(
        title=f"{res.ticker}: stock vs. market excess returns ({res.freq.lower()})",
        xaxis_title="S&P 500 excess return (%)",
        yaxis_title=f"{res.ticker} excess return (%)",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=60),
    )
    return fig


def beta_r2_bars(df: pd.DataFrame) -> go.Figure:
    """df columns: ticker, beta, r2. Adds the S&P 500 (beta = 1, R² = 1) as a reference bar."""
    names = list(df["ticker"]) + ["S&P 500"]
    betas = list(df["beta"]) + [1.0]
    r2s = list(df["r2"]) + [1.0]
    colors = ["#ef4444" if b > 1 else "#3b82f6" for b in df["beta"]] + ["#6b7280"]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Beta (systematic risk)", "R² (model reliability)"))
    fig.add_trace(go.Bar(x=names, y=betas, marker_color=colors, text=[f"{b:.2f}" for b in betas],
                         textposition="outside", name="Beta", showlegend=False), row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=r2s, marker_color=colors, text=[f"{r:.2f}" for r in r2s],
                         textposition="outside", name="R²", showlegend=False), row=1, col=2)
    fig.update_yaxes(range=[0, max(1.2, max(betas) * 1.2)], row=1, col=1)
    fig.update_yaxes(range=[0, 1.15], row=1, col=2)
    fig.update_layout(margin=dict(t=60))
    return fig


def sml_chart(df: pd.DataFrame, rf: float, mrp: float) -> go.Figure:
    """Beta (x) vs return (y) with the CAPM Security Market Line."""
    bmax = max(1.5, df["beta"].max() * 1.1)
    bmin = min(0.0, df["beta"].min() * 1.1)
    xs = np.array([bmin, bmax])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=(rf + xs * mrp) * 100, mode="lines", name="Security Market Line (CAPM)",
                             line=dict(color="#111827", width=2)))
    fig.add_trace(go.Scatter(
        x=df["beta"], y=df["realized_return"] * 100, mode="markers+text", text=df["ticker"],
        textposition="top center", name="Realized return",
        marker=dict(size=13, color=[REC_COLORS[r] for r in df["recommendation"]], line=dict(width=1, color="white")),
        hovertemplate="%{text}<br>β=%{x:.2f}<br>Realized: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["beta"], y=df["expected_return"] * 100, mode="markers", name="CAPM expected return",
        marker=dict(size=9, symbol="diamond-open", color="#111827"),
        hovertemplate="β=%{x:.2f}<br>Expected: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=[1.0], y=[(rf + mrp) * 100], mode="markers+text", text=["S&P 500"],
                             textposition="bottom right", name="Market", marker=dict(size=14, symbol="star", color="#6b7280")))
    fig.update_layout(
        title="Beta vs. annualized return (marker color = recommendation; gap above/below the line ≈ alpha)",
        xaxis_title="Beta", yaxis_title="Annualized return (%)",
        legend=dict(orientation="h", y=-0.2), margin=dict(t=70),
    )
    return fig


def cml_chart(df: pd.DataFrame, rf: float, mkt_return: float, mkt_vol: float) -> go.Figure:
    """Total risk (x) vs return (y) with the Capital Market Line through rf and the market portfolio."""
    xmax = max(df["vol_annual"].max(), mkt_vol) * 1.15
    xs = np.array([0.0, xmax])
    slope = (mkt_return - rf) / mkt_vol

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs * 100, y=(rf + slope * xs) * 100, mode="lines", name="Capital Market Line",
                             line=dict(color="#111827", width=2)))
    fig.add_trace(go.Scatter(
        x=df["vol_annual"] * 100, y=df["realized_return"] * 100, mode="markers+text", text=df["ticker"],
        textposition="top center", name="Stocks",
        marker=dict(size=13, color=[REC_COLORS[r] for r in df["recommendation"]], line=dict(width=1, color="white")),
    ))
    fig.add_trace(go.Scatter(x=[mkt_vol * 100], y=[mkt_return * 100], mode="markers+text", text=["S&P 500"],
                             textposition="bottom right", name="Market portfolio",
                             marker=dict(size=14, symbol="star", color="#6b7280")))
    fig.add_trace(go.Scatter(x=[0], y=[rf * 100], mode="markers+text", text=["Risk-free"],
                             textposition="top right", name="Risk-free", marker=dict(size=10, color="#2563eb")))
    fig.update_layout(
        title="Total volatility vs. annualized return (individual stocks sit below the CML: idiosyncratic risk is unrewarded)",
        xaxis_title="Annualized volatility (%)", yaxis_title="Annualized return (%)",
        legend=dict(orientation="h", y=-0.2), margin=dict(t=70),
    )
    return fig


def residuals_vs_fitted(res: CAPMResult) -> go.Figure:
    fig = go.Figure(go.Scatter(x=res.model.fittedvalues * 100, y=res.model.resid * 100, mode="markers",
                               marker=dict(opacity=0.6, color="#2563eb")))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Residuals vs. fitted values", xaxis_title="Fitted excess return (%)",
                      yaxis_title="Residual (%)", margin=dict(t=60))
    return fig


def rolling_beta_chart(rb: pd.Series, full_beta: float, ticker: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=rb.index, y=rb.values, mode="lines", name="Rolling beta", line=dict(color="#2563eb")))
    fig.add_hline(y=full_beta, line_dash="dash", line_color="#dc2626", annotation_text=f"Full-sample β = {full_beta:.2f}")
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
    fig.update_layout(title=f"{ticker}: rolling beta", yaxis_title="Beta", margin=dict(t=60))
    return fig
