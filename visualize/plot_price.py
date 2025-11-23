import plotly.graph_objects as go
from visualize.style_figure import style_figure

def plot_price(price_fc, now_floor):
    
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
    if "predicted_price(50%)" in price_fc.columns:
        fig_p.add_trace(
            go.Scatter(
                x=price_fc["timestamp"],
                y=price_fc["predicted_price(50%)"],
                mode="lines",
                name="予測価格",
                line=dict(width=2, color="#FFCA28", dash="dot"),  # 点線
            )
        )

    # 分位点バンド（10〜90）
    if all(c in price_fc.columns for c in ["predicted_price(10%)", "predicted_price(90%)"]):
        fig_p.add_trace(
            go.Scatter(
                x=price_fc["timestamp"],
                y=price_fc["predicted_price(90%)"],
                mode="lines",
                name="P90",
                line=dict(width=0, color="#FA9A8D"),
                showlegend=False,
            )
        )
        fig_p.add_trace(
            go.Scatter(
                x=price_fc["timestamp"],
                y=price_fc["predicted_price(10%)"],
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
        y_dtick=1000,
    )
    fig_p.update_xaxes(tickangle=-45)
    
    return fig_p