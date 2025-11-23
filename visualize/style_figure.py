# テーマに合わせた色
PLOT_BG_COLOR = "#0b1633"   # グラフ内の背景（今より少し明るい紺）
PAPER_BG_COLOR = "#020922"  # グラフ外側（アプリ背景と同じでOK）
GRID_COLOR = "#2c3f66"      # グリッド
TEXT_COLOR = "#f5f5f5"      # 文字色

def style_figure(
    fig,
    x_title: str | None = None,
    y_title: str | None = None,
    x_dtick=None,
    y_dtick=None,
):
    """
    Plotly 図に共通のスタイルを当てる。
    - 背景色
    - グリッド色
    - フォントサイズ
    - 軸タイトル
    - 目盛り間隔 (dtick)
    """
    fig.update_layout(
        paper_bgcolor=PAPER_BG_COLOR,
        plot_bgcolor=PLOT_BG_COLOR,
        font=dict(color=TEXT_COLOR, size=14),  # 全体のデフォルト文字サイズ
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        tickfont=dict(size=12),       # x 軸の目盛りの文字
        title_font=dict(size=16),     # x 軸タイトル
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        tickfont=dict(size=12),       # y 軸の目盛りの文字
        title_font=dict(size=16),     # y 軸タイトル
    )

    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)

    # 目盛りの細かさ（任意）
    if x_dtick is not None:
        fig.update_xaxes(dtick=x_dtick)
    if y_dtick is not None:
        fig.update_yaxes(dtick=y_dtick)

    return fig
