import plotly.graph_objects as go
from visualize.style_figure import style_figure

def plot_demand(demand, now_floor):
    fig_d = go.Figure()
    if "realized_demand" in demand.columns:
        fig_d.add_trace(
            go.Scatter(
                x=demand["timestamp"],
                y=demand["realized_demand"],
                mode="lines",
                name="需要実績",
                line=dict(width=2.5, color="#4FC3F7"),  # 明るい水色
            )
        )

    if "predicted_demand" in demand.columns:
        fig_d.add_trace(
            go.Scatter(
                x=demand["timestamp"],
                y=demand["predicted_demand"],
                mode="lines",
                name="需要予測",
                line=dict(width=2, color="#FFCA28", dash="dot"),  # 点線
            )
        )

    # 現在時刻の縦線
    fig_d.add_shape(
        type="line",
        x0=now_floor, x1=now_floor,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="red", dash="dot")  # 赤色＋破線
    )

    # 注釈を追加して「Now」と表示
    fig_d.add_annotation(
        x=now_floor,
        y=1.05,  # yref="paper" の上端に近い位置
        xref="x", yref="paper",
        text="Now",
        showarrow=False,
        font=dict(color="red", size=12),
        align="center"
    )


    six_hours_ms = 6 * 60 * 60 * 1000  # 6時間ごとに縦グリッド
    one_day_ms = 24 * 60 * 60 * 1000 # 1日ごと
    
    fig_d = style_figure(
        fig_d,
        x_title="日時",
        y_title="電力需要 [MW]",
        x_dtick=one_day_ms,
        x_minor_dtick=six_hours_ms,  # グリッドを細かく
        y_dtick=2000,
    )

    fig_d.update_traces(
        # ホバーのヘッダー部分（時間）の形式を指定
        hovertemplate="<b>%{x|%Y/%m/%d %H:%M}</b><br>" + 
                    "%{fullData.name}: %{y:,.0f} MW<extra></extra>"
    )
    
    return fig_d