import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualize.style_figure import style_figure

COLOR_MAP = {
    "lng":           "#F4A6A6",  # パステルレッド（LNG）
    "coal":          "#F7C1C1",  # パステルピンク（石炭）
    "oil":           "#FFD6E8",  # 明るいピンク（石油）
    "th_other":      "#FFEAF2",  # ごく淡いピンク（火力その他）
    "biomass":       "#D9F9B1",  # パステル黄緑（バイオマス）
    "wind_net":      "#B7E4B6",  # パステルグリーン（風力）
    "hydro":         "#B3D9FF",  # パステルブルー（水力）
    "pstorage_gen":  "#CDEEFF",  # 淡い水色（揚水発電）
    "pv_net":        "#FFFACD",  # レモンシフォン（太陽光）
    "battery_dis":   "#FFDAB3",  # パステルオレンジ（蓄電池放電）
    "import":        "#D3D3D3",  # ライトグレー（連系線受電）
    "misc":          "#D8BFD8",  # パステルパープル（その他）
}

def plot_energy_mix(dispatch_display: pd.DataFrame, dispatch_error: pd.DataFrame, now_floor):
    df = dispatch_display.copy()
    err = dispatch_error.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    err["timestamp"] = pd.to_datetime(err["timestamp"])

    df = df.sort_values("timestamp")
    err = err.sort_values("timestamp")

    num_cols = [
        "pv", "wind", "hydro", "coal", "oil", "lng", "th_other", "biomass", "misc",
        "curtail_pv", "curtail_wind",
        "pstorage_gen", "pstorage_pump",
        "battery_dis", "battery_ch",
        "import",
        "reserve", "shed", "predicted_demand", "total_cost"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["pv_net"] = np.maximum(df.get("pv", 0) - df.get("curtail_pv", 0), 0)
    df["wind_net"] = np.maximum(df.get("wind", 0) - df.get("curtail_wind", 0), 0)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3],
        subplot_titles=("エネルギーミックス", "ネットエラー"),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    stack_order = [
        "lng", "coal", "oil", "th_other", "biomass",
        "wind_net", "hydro", "pstorage_gen", "pv_net",
        "battery_dis", "import", "misc"
    ]
    name_map = {
        "lng": "LNG",
        "coal": "石炭",
        "oil": "石油",
        "th_other": "火力(その他)",
        "biomass": "バイオマス",
        "wind_net": "風力",
        "hydro": "水力",
        "pstorage_gen": "揚水(発電)",
        "pv_net": "太陽光",
        "battery_dis": "蓄電池(放電)",
        "import": "連系線",
        "misc": "その他",
    }

    for col in stack_order:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[col],
                    mode="lines",
                    stackgroup="supply",
                    fill="tonexty",
                    name=name_map.get(col, col),
                    line=dict(width=0.5, color=COLOR_MAP.get(col, None)),
                    fillcolor=COLOR_MAP.get(col, None)
                ),
                row=1, col=1, secondary_y=False
            )

    demand_col = "predicted_demand" if "predicted_demand" in df.columns else None
    if demand_col is not None:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[demand_col],
                name="需要",
                mode="lines",
                line=dict(color="#4FC3F7", width=2.5),
            ),
            row=1, col=1, secondary_y=False
        )

    if "total_cost" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["total_cost"],
                name="総コスト",
                mode="lines",
                line=dict(width=2.5, color="#FFCA28", dash="dot"),
            ),
            row=1, col=1, secondary_y=True
        )

    error_cols = [c for c in err.columns if c.endswith("_error")]
    err_name_map = {
        "lng_error": "LNG", "coal_error": "石炭", "oil_error": "石油",
        "th_other_error": "火力(その他)", "biomass_error": "バイオマス",
        "wind_error": "風力", "hydro_error": "水力",
        "pstorage_gen_error": "揚水(発電)", "pv_error": "太陽光",
        "battery_dis_error": "蓄電池(放電)", "import_error": "連系線",
        "misc_error": "その他", "curtail_pv_error": "PV抑制",
        "curtail_wind_error": "風力抑制",
    }
    color_map = {
    "lng_error": "#F4A6A6",
    "coal_error": "#F7C1C1",
    "oil_error": "#FFD6E8",
    "th_other_error": "#FFEAF2",
    "biomass_error": "#D9F9B1",
    "wind_error": "#B7E4B6",
    "hydro_error": "#B3D9FF",
    "pstorage_gen_error": "#CDEEFF",
    "pv_error": "#FFFACD",
    "battery_dis_error": "#FFDAB3",
    "import_error": "#D3D3D3",
    "misc_error": "#D8BFD8",
    "curtail_pv_error": "#FFB3B3",
    "curtail_wind_error": "#C6F6C6",
}

    for col in error_cols:
        fig.add_trace(
            go.Bar(
                x=err["timestamp"],
                y=err[col],
                name=err_name_map.get(col, col),
                marker_color=color_map.get(col, "#999999"),
                showlegend=False,
            ),
            row=2, col=1
        )

    for r in [1, 2]:
        fig.add_vline(
            x=now_floor,
            line_color="red",
            line_dash="dot",
            row=r,
            col=1,
        )

    fig.add_annotation(
        x=now_floor,
        y=1.02,
        xref="x",
        yref="paper",
        text="Now",
        showarrow=False,
        font=dict(color="red", size=12),
    )

    one_day_ms = 24 * 60 * 60 * 1000
    one_hour_ms = 60 * 60 * 60 * 1000 / 60  # = 1 hour in ms
    fig = style_figure(
        fig,
        x_title="日時",
        y_title="",
        x_dtick=one_day_ms,
        x_minor_dtick=one_hour_ms,
        x_is_datetime=True,
    )

    fig.update_yaxes(title_text="電力 [MW]", row=1, col=1, secondary_y=False, dtick=2000, tickformat="~s")
    fig.update_yaxes(title_text="コスト", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="ネットエラー [MW]", row=2, col=1, dtick=2000, tickformat="~s")

    fig.update_layout(
        height=760,
        hovermode="x unified",
        barmode="relative",
        margin=dict(l=60, r=40, t=120, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
        ),
        bargap=0.1,
    )

    fig.update_traces(
        hovertemplate="<b>%{x|%Y/%m/%d %H:%M}</b><br>%{fullData.name}: %{y:,.0f} MW<extra></extra>"
    )

    return fig