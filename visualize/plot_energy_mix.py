import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

def plot_energy_mix(out_df: pd.DataFrame, now_floor):
    df = out_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # 数値にしておく（存在する列だけ）
    num_cols = [
        "pv","wind","hydro","coal","oil","lng","th_other", "biomass", "misc"
        "curtail_pv","curtail_wind",
        "pstorage_gen","pstorage_pump",
        "battery_dis","battery_ch",
        "import",
        "reserve","shed","predicted_demand","total_cost"
    ]
    for c in num_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # --- 純発電（抑制を引いた分） ---
    df["pv_net"]   = np.maximum(df.get("pv", 0)   - df.get("curtail_pv", 0), 0)
    df["wind_net"] = np.maximum(df.get("wind", 0) - df.get("curtail_wind", 0), 0)

    # --- バランス残差チェック ---
    # モデルの需給式：
    #  PV + 風 + 水力 + 火力 + 揚水発電 + 電池放電 + 連系線 + Shed
    #    = 需要 + 揚水揚水 + 電池充電
    gen_total = (
        df.get("pv", 0)
        + df.get("wind", 0)
        + df.get("hydro", 0)
        + df.get("coal", 0)
        + df.get("oil", 0)
        + df.get("lng", 0)
        + df.get("th_other", 0)
        + df.get("biomass", 0)
        + df.get("misc", 0)
        + df.get("pstorage_gen", 0)
        + df.get("battery_dis", 0)
        + df.get("import", 0)
    )
    demand_total = (
        df.get("load_MW", 0)
        + df.get("pstorage_pump", 0)
        + df.get("battery_ch", 0)
    )
    resid = gen_total + df.get("shed", 0) - demand_total
    max_abs_resid = float(np.abs(resid).max())
    if max_abs_resid > 50:  # 50MW超えなら警告
        print(f"[warn] balance residual max |resid| = {max_abs_resid:.1f} MW (確認推奨)")

    fig = go.Figure()

    # --- 積み上げ（供給側：純発電＋揚水発電＋電池放電＋連系線） ---
    stack = "netgen"
    def add_stack(col, name):
        if col in df and df[col].abs().sum() > 0:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[col],
                name=name, mode="lines",
                stackgroup=stack,
                line=dict(width=0.0),
                fillcolor=COLOR_MAP.get(col, "#cccccc")  # fallback色

            ))

    # 下から積む順序（ベースに近いものを下に）
    add_stack("pstorage_gen",  "揚水(発電)")
    add_stack("hydro",         "水力")
    add_stack("oil",           "石油")
    add_stack("coal",          "石炭")
    add_stack("lng",           "LNG")
    add_stack("pv_net",        "太陽光(純)")
    add_stack("battery_dis",   "蓄電池(放電)")
    add_stack("wind_net",      "風力(純)")
    add_stack("import",        "連系線(流入)")
    add_stack("th_other",      "火力(その他)")
    add_stack("biomass",       "バイオマス")
    add_stack("misc",          "その他")

    # --- 需要ライン（本来の需要） ---
    if "predicted_demand" in df:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["predicted_demand"], name="需要（予測）",
            mode="lines", line=dict(color="#4FC3F7", width=2.5)
        ))

    # --- 予備力帯（load → load+reserve を塗る） ---
    if "reserve" in df and df["reserve"].abs().sum() > 0:
        # 下側（需要）をダミーで描く
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["predicted_demand"],
            showlegend=False, line=dict(width=0), hoverinfo="skip"
        ))
        # 上側（需要＋予備力）
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["predicted_demand"] + df["reserve"],
            name="予備力帯",
            fill="tonexty", fillcolor="rgba(255,165,0,0.15)",
            line=dict(width=0),
            hovertemplate="%{x|%H:%M}<br>予備力: %{customdata:.0f} MW",
            customdata=df["reserve"]
        ))

    # --- 抑制は別レイヤ（純発電の上に薄く重ねる） ---
    if "curtail_pv" in df and df["curtail_pv"].sum() > 0:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["pv_net"] + df["curtail_pv"],
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["pv_net"],
            name="出力抑制(PV)",
            fill="tonexty", fillcolor="rgba(255,0,0,0.12)",
            line=dict(width=0),
            hovertemplate="%{x|%H:%M}<br>PV抑制: %{customdata:.0f} MW",
            customdata=df["curtail_pv"]
        ))

    if "curtail_wind" in df and df["curtail_wind"].sum() > 0:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["wind_net"] + df["curtail_wind"],
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["wind_net"],
            name="出力抑制(風力)",
            fill="tonexty", fillcolor="rgba(0,0,255,0.12)",
            line=dict(width=0),
            hovertemplate="%{x|%H:%M}<br>風力抑制: %{customdata:.0f} MW",
            customdata=df["curtail_wind"]
        ))

    # --- 供給不足ライン（実際に賄えた需要＝需要 - Shed） ---
    if "shed" in df and df["shed"].max() > 0:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["predicted_demand"] - df["shed"],
            name="供給不足(実供給)",
            mode="lines",
            line=dict(color="crimson", dash="dot"),
            hovertemplate="%{x|%H:%M}<br>不足: %{customdata:.0f} MW",
            customdata=df["shed"]
        ))

    # --- 総コストは右軸 ---
    if "total_cost" in df:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["total_cost"],
            name="総コスト",
            mode="lines",
            line=dict(width=2.5, color="#FFCA28", dash="dot"),
            yaxis="y2"
        ))
    
    # 現在時刻の縦線
    fig.add_shape(
        type="line",
        x0=now_floor, x1=now_floor,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="red", dash="dot")  # 赤色＋破線
    )

    # 注釈を追加して「Now」と表示
    fig.add_annotation(
        x=now_floor,
        y=1.05,  # yref="paper" の上端に近い位置
        xref="x", yref="paper",
        text="Now",
        showarrow=False,
        font=dict(color="red", size=12),
        align="center"
    )  

    one_hours_ms = 60 * 60 * 1000  # 1時間ごとに縦グリッド
    fig = style_figure(
        fig,
        x_title="時刻",
        y_title="電力 [MW]",
        x_dtick=one_hours_ms,  # グリッドを細かく
        y_dtick=1000,
    )

    # そのあとで、このグラフ固有のレイアウトを上書き
    fig.update_layout(
        # title=title,
        yaxis2=dict(title="コスト", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        margin=dict(l=60, r=60, t=60, b=40),
        height=560,
    )

    return fig
