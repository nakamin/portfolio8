# テーマに合わせた色
PLOT_BG_COLOR = "#0b1633"   # グラフ内の背景（今より少し明るい紺）
PAPER_BG_COLOR = "#020922"  # グラフ外側（アプリ背景と同じでOK）
GRID_MINOR_COLOR = "#2c3f66"      # グリッド
GRID_COLOR = "#4e6691"
TEXT_COLOR = "#f5f5f5"      # 文字色
TEXT_MINOR_COLOR = "#9eb3d4"

def style_figure(
    fig,
    x_title: str | None = None,
    y_title: str | None = None,
    x_dtick=None,
    x_minor_dtick=None,
    y_dtick=None,
):
    """
    Plotly 図に共通のスタイルを当てる。
    - 背景色
    - グリッド色
    - フォントサイズ
    - 軸タイトル
    - 目盛り間隔 (dtick)
    - 凡例
    """
    fig.update_layout(
        paper_bgcolor=PAPER_BG_COLOR,
        plot_bgcolor=PLOT_BG_COLOR,
        font=dict(color=TEXT_COLOR, size=14),  # 全体のデフォルト文字サイズ
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(font_size=13),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_MINOR_COLOR,
        gridwidth=1,
        dtick=x_minor_dtick,
        zeroline=False,
        tickfont=dict(size=11, color=TEXT_MINOR_COLOR), # x 軸の目盛りの文字
        title_font=dict(size=16), # x 軸タイトル
        tickangle=0,
        tickformat="%m/%d\n%H:%M",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_MINOR_COLOR,
        zeroline=False,
        tickfont=dict(size=12),       # y 軸の目盛りの文字
        title_font=dict(size=16),     # y 軸タイトル
    )

    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    
    # 補助目盛
    if x_minor_dtick is not None:
            fig.update_xaxes(
                minor=dict(
                    dtick=x_minor_dtick,
                    gridcolor=GRID_MINOR_COLOR,
                    gridwidth=1,
                    showgrid=True
                )
            )
        
    if x_dtick is not None:
        fig.update_xaxes(
                dtick=x_dtick,      # 24時間おき
                gridcolor=GRID_COLOR, # 00:00用の明るい色
                gridwidth=2,        # ここで線を太くする
                showgrid=True
        )
        
    if y_dtick is not None:
        fig.update_yaxes(dtick=y_dtick)
    
    return fig
