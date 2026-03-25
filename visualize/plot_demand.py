import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualize.style_figure import style_figure

def plot_demand_with_error(demand_df, now_floor):
    df = demand_df.copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("電力需要", "予測誤差")
    )

    # 実績
    realized = df.dropna(subset=["realized_demand"])
    fig.add_trace(
        go.Scatter(
            x=realized["timestamp"],
            y=realized["realized_demand"],
            mode="lines",
            name="需要実績",
            line=dict(width=2.5, color="#4FC3F7"),
        ),
        row=1, col=1
    )

    # # 過去予測
    # past_pred = df.dropna(subset=["predicted_demand_past"])
    # fig.add_trace(
    #     go.Scatter(
    #         x=past_pred["timestamp"],
    #         y=past_pred["predicted_demand_past"],
    #         mode="lines",
    #         name="需要予測",
    #         line=dict(width=2, color="#FFCA28", dash="dot"),
    #     ),
    #     row=1, col=1
    # )

    # 未来予測
    future_pred = df.dropna(subset=["predicted_demand_future_past"])
    fig.add_trace(
        go.Scatter(
            x=future_pred["timestamp"],
            y=future_pred["predicted_demand_future_past"],
            mode="lines",
            name="需要予測",
            line=dict(width=2, color="#FFCA28", dash="dash"),
        ),
        row=1, col=1
    )

    # 下段 誤差
    err = df.dropna(subset=["abs_error"])
    fig.add_trace(
        go.Bar(
            x=err["timestamp"],
            y=err["abs_error"],
            name="| 予測 - 実績 |",
            marker_color="#FFCA28",
        ),
        row=2, col=1
    )

    # Now線
    for r in [1, 2]:
        fig.add_vline(
            x=now_floor,
            line_color="red",
            line_dash="dot",
            row=r, col=1
        )

    fig.update_yaxes(title_text="需要 [MW]", row=1, col=1, tickformat="~s")
    fig.update_yaxes(title_text="誤差 [MW]", row=2, col=1, tickformat="~s")

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