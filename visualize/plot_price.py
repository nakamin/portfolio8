import pandas as pd
import plotly.graph_objects as go
from visualize.style_figure import style_figure

def add_band(fig, df, name, color, opacity):
    # 上側（P90）
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["predicted_price(90%)"],
            mode="lines",
            line=dict(width=0, color=color),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # 下側（P10）を塗りつぶし（P90→P10の順で塗る）
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["predicted_price(10%)"],
            mode="lines",
            fill="tonexty",
            name=name,
            line=dict(width=0, color=color),
            opacity=opacity,
        )
    )
    return fig


def plot_price(price_fc, today, now_floor):
    
    tomorrow0 = pd.Timestamp(today) + pd.Timedelta(days=1)
    cut_band = tomorrow0 + pd.Timedelta(days=1)             # 明後日0:00
    
    mask_day1 = (price_fc["timestamp"] >= tomorrow0) & (price_fc["timestamp"] < cut_band)
    mask_day2plus = (price_fc["timestamp"] >= cut_band)

    df1 = price_fc.loc[mask_day1].copy()
    df2 = price_fc.loc[mask_day2plus].copy()

    fig_p = go.Figure()

    # 実績価格
    if "tokyo_price_jpy_per_kwh" in price_fc.columns:
        fig_p.add_trace(
            go.Scatter(
                x=price_fc["timestamp"],
                y=price_fc["tokyo_price_jpy_per_kwh"],
                mode="lines",
                name="スポット価格（実績）",
                line=dict(width=2.5, color="#4FC3F7"),  # 明るい水色
            )
        )

    # P50（中心予測）
    if "predicted_price(50%)" in df1.columns:
        fig_p.add_trace(
            go.Scatter(
                x=df1["timestamp"],
                y=df1["predicted_price(50%)"],
                mode="lines",
                name="予測価格",
                line=dict(width=2, color="#FFCA28", dash="dot"),  # 点線
            )
        )
        
    # fig_p = add_band(fig_p, df1, name="P10–P90（+24hまで）", color="#FA9A8D", opacity=0.20)
    # fig_p = add_band(fig_p, df2, name="P10–P90（+48h以降は参考）", color="#FA9A8D", opacity=0.10)

    # 分位点バンド（10〜90）
    if all(c in df1.columns for c in ["predicted_price(10%)", "predicted_price(90%)"]):
        fig_p.add_trace(
            go.Scatter(
                x=df1["timestamp"],
                y=df1["predicted_price(90%)"],
                mode="lines",
                name="P90",
                line=dict(width=0, color="#FA9A8D"),
                showlegend=False,
            )
        )
        fig_p.add_trace(
            go.Scatter(
                x=df1["timestamp"],
                y=df1["predicted_price(10%)"],
                mode="lines",
                name="P10-P90",
                fill="tonexty",  # P90 と P10 の間を塗る
                line=dict(width=0, color="#FA9A8D"),
                opacity=0.2,
                showlegend=True,
            )
        )

    # 現在時刻の縦線
    fig_p.add_shape(
        type="line",
        x0=now_floor, x1=now_floor,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="red", dash="dot")  # 赤色＋破線
    )

    # 注釈を追加して「Now」と表示
    fig_p.add_annotation(
        x=now_floor,
        y=1.05,  # yref="paper" の上端に近い位置
        xref="x", yref="paper",
        text="Now",
        showarrow=False,
        font=dict(color="red", size=12),
        align="center"
    )

    six_hours_ms = 6 * 60 * 60 * 1000  # 6時間ごとに縦グリッド
    fig_p = style_figure(
        fig_p,
        x_title="日時",
        y_title="価格 [円/kWh]",
        x_dtick=six_hours_ms,  # グリッドを細かく
        y_dtick=10,
    )
    fig_p.update_xaxes(tickangle=-45)
    
    return fig_p