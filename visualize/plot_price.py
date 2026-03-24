import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_price_with_crps(price_df, now_floor):
    df = price_df.copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("JEPX東京スポット価格予測", "CRPS")
    )

    realized = df.dropna(subset=["tokyo_price_jpy_per_kwh"])
    fig.add_trace(
        go.Scatter(
            x=realized["timestamp"],
            y=realized["tokyo_price_jpy_per_kwh"],
            mode="lines",
            name="スポット価格（実績）",
            line=dict(width=2.5, color="#4FC3F7"),
        ),
        row=1, col=1
    )

    # past_pred = df.dropna(subset=["predicted_price_past(50%)"])
    # fig.add_trace(
    #     go.Scatter(
    #         x=past_pred["timestamp"],
    #         y=past_pred["predicted_price_past(50%)"],
    #         mode="lines",
    #         name="価格予測（P50）",
    #         line=dict(width=2, color="#FFCA28", dash="dot"),
    #     ),
    #     row=1, col=1
    # )

    future_pred = df.dropna(subset=["predicted_price_future_past(50%)"])
    fig.add_trace(
        go.Scatter(
            x=future_pred["timestamp"],
            y=future_pred["predicted_price_future(90%)"],
            mode="lines",
            line=dict(width=0, color="#FA9A8D"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=future_pred["timestamp"],
            y=future_pred["predicted_price_future(10%)"],
            mode="lines",
            fill="tonexty",
            name="価格予測 P10-P90",
            line=dict(width=0, color="#FA9A8D"),
            opacity=0.2,
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=future_pred["timestamp"],
            y=future_pred["predicted_price_future_past(50%)"],
            mode="lines",
            name="価格予測（P50）",
            line=dict(width=2, color="#FFCA28", dash="dash"),
        ),
        row=1, col=1
    )

    crps_df = df.dropna(subset=["crps"])
    fig.add_trace(
        go.Bar(
            x=crps_df["timestamp"],
            y=crps_df["crps"],
            name="CRPS",
            marker_color="#FA9A8D",
        ),
        row=2, col=1
    )

    for r in [1, 2]:
        fig.add_vline(
            x=now_floor,
            line_color="red",
            line_dash="dot",
            row=r, col=1
        )

    fig.update_yaxes(title_text="価格 [円/kWh]", row=1, col=1)
    fig.update_yaxes(title_text="CRPS", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=700,
        margin=dict(l=60, r=40, t=80, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        bargap=0.1,
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        tickformat="%m/%d\n%H:%M",
        dtick=24 * 60 * 60 * 1000,
        minor=dict(dtick=6 * 60 * 60 * 1000, showgrid=True),
        row=2, col=1
    )

    return fig