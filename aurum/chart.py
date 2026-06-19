import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def generate_chart(df: pd.DataFrame, tp1: float, tp2: float, sl: float, signal: str) -> go.Figure:
    plot_df = df.tail(120)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
    )

    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="GC=F",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["EMA_9"], line=dict(color="#FFD700", width=1), name="EMA 9"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["EMA_50"], line=dict(color="#00FF66", width=1.5), name="EMA 50"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["POC"], line=dict(color="#FF00FF", width=1, dash="dot"), name="POC"),
        row=1,
        col=1,
    )

    if signal in ("BUY", "SELL"):
        fig.add_hline(y=tp1, line_dash="dash", line_color="#00FF66", annotation_text=f"TP1: {tp1}", row=1, col=1)
        fig.add_hline(y=tp2, line_dash="dash", line_color="#00DDFF", annotation_text=f"TP2: {tp2}", row=1, col=1)
        fig.add_hline(y=sl, line_dash="dash", line_color="#FF3333", annotation_text=f"SL: {sl}", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["RSI"], line=dict(color="#E5E9F0", width=1.5), name="RSI"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["RSI_Overbought"], line=dict(color="#FF3333", dash="dash"), name="RSI OB"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["RSI_Oversold"], line=dict(color="#00FF66", dash="dash"), name="RSI OS"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["Corr_DXY_Z"], line=dict(color="#FF9900"), name="Z-Corr DXY"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["Corr_US10Y_Z"], line=dict(color="#3399FF"), name="Z-Corr US10Y"),
        row=3,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=750,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def generate_equity_chart(equity: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(equity))), y=equity, fill="tozeroy", line=dict(color="#FFD700")))
    fig.update_layout(
        template="plotly_dark",
        title="Equity Curve (Backtest)",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Сделки",
        yaxis_title="Капитал ($)",
    )
    return fig
