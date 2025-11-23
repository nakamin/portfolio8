from pathlib import Path
from datetime import datetime, timedelta
import json

import pandas as pd
import pytz
import streamlit as st

from visualize.plot_energy_mix import plot_energy_mix
from visualize.plot_price import plot_price
from visualize.plot_demand import plot_demand

# ========================
# 基本設定
# ========================

CACHE_DIR = Path("data/cache")
STATIC_DIR = Path("data/static")
JST = pytz.timezone("Asia/Tokyo")

st.set_page_config(
    layout="wide",
)

# ========================
# 共通ユーティリティ
# ========================

@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    path = CACHE_DIR / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def jst_now_floor_30min() -> datetime:
    """現在時刻をJSTで30分刻みに丸める"""
    now = datetime.now(JST)
    minute = 0 if now.minute < 30 else 30
    return now.replace(minute=minute, second=0, microsecond=0)

# ========================
# ヘッダー（画像・説明・現在時刻）
# ========================

hero_path = STATIC_DIR / "hero.png"
if hero_path.exists():
    st.image(str(hero_path), use_container_width=True, width=800)

now_jst = datetime.now(JST)
st.caption(f"現在時刻（JST）: {now_jst.strftime('%Y-%m-%d %H:%M')}")

meta = load_metadata()
if meta.get("last_updated_utc"):
    last_utc = datetime.fromisoformat(meta["last_updated_utc"])
    last_jst = last_utc.astimezone(JST)
    st.caption(f"データ更新時刻: {last_jst.strftime('%Y-%m-%d %H:%M')}（自動更新日時）")
else:
    st.caption("データ更新時刻: 不明")

now_floor = jst_now_floor_30min()
today = now_floor.date()
demand_from = today
demand_to = today + timedelta(days=6)  # 今日を含めて7日間
price_from = today + timedelta(days=1)  # 明日スタート
price_to = today + timedelta(days=6)    # 6日後まで（6日間）

tab_dashboard, tab_model, tab_pipeline, contact = st.tabs(["ダッシュボード", "モデル説明", "パイプライン説明", "お問い合わせ"])

with tab_dashboard:
    # ========================
    # セクション1: 需要予測
    # ========================

    st.subheader("1. 電力需要（実績＋予測）")

    demand = load_parquet("demand_forecast")      # predicted_demand / realized_demand / demand など
    print("demand: \n", demand)

    fig_d = plot_demand(demand, now_floor)
    st.plotly_chart(fig_d, use_container_width=True)
    st.markdown(
    f"""
    <h4 style="text-align:center; margin-top: 1rem;">
      予測対象期間：{demand_from:%Y年%m月%d日} 〜 {demand_to:%Y年%m月%d日}（7日間）
    </h4>
    """,
    unsafe_allow_html=True,
)

    st.markdown("---")

    # ========================
    # セクション2: 価格（実績＋分位点予測）予測
    # ========================

    st.subheader("2. JEPX 東京スポット価格と分位点予測")

    price_fc = load_parquet("price_forecast")          # price / predicted_price(10/50/90) / tokyo_price_jpy_per_kwh
    print("price_fc: \n", price_fc)

    fig_p = plot_price(price_fc=price_fc, now_floor=now_floor)
    st.plotly_chart(fig_p, use_container_width=True)
    
    st.markdown(
        f"""
        <h4 style="text-align:center; margin-top: 1rem;">
        予測対象期間：{price_from:%Y年%m月%d日} 〜 {price_to:%Y年%m月%d日}（6日間）
        </h4>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ========================
    # セクション3: 発電ミックス最適化結果
    # ========================

    st.subheader("3. エネルギーミックス（最適化結果）")

    dispatch = load_parquet("dispatch_optimal")
    print("dispatch: \n", dispatch)

    fig_balance = plot_energy_mix(dispatch, now_floor)
    st.plotly_chart(fig_balance, use_container_width=True)
    
    st.markdown(
    f"""
    <h4 style="text-align:center; margin-top: 1rem;">
      予測対象日：{today:%Y年%m月%d日}
    </h4>
    """,
    unsafe_allow_html=True,
)


with tab_model:
    sources = meta.get("sources", {})
    
    st.markdown("### 需要予測モデル")
    st.markdown(
        f"""
- 特徴量: 過去需要、気象（気温・日射・風速）、休日フラグ など
- モデル: {sources.get("demand_model", "不明")}
- 目的: 30分先〜7日先までの需要分布（P10 / P50 / P90）を推定
        """
    )

    st.markdown("### 価格予測モデル")
    st.markdown(
        f"""
- 特徴量: 需要予測、燃料価格（Brent / Henry Hub / 石炭）、為替、天候など
- モデル: {sources.get("price_model", "不明")}
- 目的: 東京エリア JEPX スポット価格のリスクレンジを推定
        """
    )

    st.markdown("### 最適化モデル")
    st.markdown(
        """
- 目的関数: 供給コスト最小化
- 制約条件:
    - 需要=供給バランス
    - 各電源の最大・最小出力
    - ランプ制約、揚水・蓄電池の入出力制約 など
        """
    )
    
with tab_pipeline:
    meta = load_metadata()
    src = meta.get("sources", {})
    markets = src.get("markets", {})

    st.markdown("### 全体フロー")
    st.markdown(
        """
1. 需要・価格・天気・燃料市況の取得（`fetch_*.py`）
2. 前処理・特徴量生成（`predict_demand.py`, `predict_price.py`）
3. 単位時間ごとのディスパッチ最適化（`optimize_dispatch.py`）
4. 結果を `data/cache/*.parquet` に保存（`run_daily_pipeline.py`）
        """
    )

    st.markdown("---")
    st.markdown("### 1. データ取得フェーズ（`fetch_*.py`）")

    st.markdown("**需要・価格**")
    st.markdown(
        f"""
- 需要実績: 電力会社公表データ（例：東京電力「でんき予報」実績値 CSV）
- 市場価格: {src.get("jepx", "JEPX 日前スポット 東京エリア")}
        """
    )

    st.markdown("**天気（気温・日射・風速 など）**")
    st.markdown(f"- {src.get('weather', 'Open-Meteo JMA API')}")

    st.markdown("**燃料市況**")
    st.markdown(
        "\n".join(
            [
                f"- 為替: {markets.get('fx', 'Frankfurter / ECB')}",
                f"- 原油: {markets.get('oil', 'EIA / yfinance BZ=F')}",
                f"- LNG: {markets.get('lng', 'EIA / yfinance NG=F')}",
                f"- 石炭: {markets.get('coal', 'World Bank Pink Sheet')}",
            ]
        )
    )

    st.markdown("---")
    st.markdown("### 2. 前処理・特徴量生成（`predict_demand.py`, `predict_price.py`）")

    st.markdown("**需要予測**")
    st.markdown(
        f"""
- 利用データ: 過去需要、気象、曜日・休日フラグ など  
- モデル: {src.get("demand_model", "自作需要予測モデル")}
        """
    )

    st.markdown("**価格予測**")
    st.markdown(
        f"""
- 利用データ: 需要予測、JEPX 過去スポット価格、為替、Brent、Henry Hub、石炭価格 など  
- モデル: {src.get("price_model", "自作価格予測モデル")}
- 出力: P10 / P50 / P90 の分位点価格
        """
    )

    st.markdown("---")
    st.markdown("### 3. ディスパッチ最適化（`optimize_dispatch.py`）")
    st.markdown(
        f"""
- 目的: 予測需要を満たしつつ総コストを最小化  
- 入力: 需要予測、電源ごとの制約、燃料コスト など  
- モデル: {src.get("optimizer", "自作数理最適化モデル")}
        """
    )

    st.markdown("---")
    st.markdown("### 4. キャッシュ・ダッシュボード連携（`run_daily_pipeline.py`）")
    st.markdown(
        """
- 上記 1〜3 を一括実行し、`data/cache/*.parquet` を生成  
- 併せて `metadata.json` に最終更新時刻・データソース・モデル情報を保存  
- このダッシュボードは `metadata.json` を読み込んで、画面上部の更新日時や
  「データ・モデル情報」タブ、フッターのクレジットに反映している
        """
    )


with contact:
    # ========================
    # セクション4: お問い合わせ
    # ========================
    st.subheader("お問い合わせフォーム")

    st.markdown(
        """
    モデルや可視化に関するご意見等がございましたら下記フォームよりお問い合わせください。
    """
    )

    def save_contact(name: str, email: str, message: str):
        contact_path = Path("data/contact_messages.csv")
        contact_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        row = pd.DataFrame(
            [[ts, name, email, message]],
            columns=["timestamp", "name", "email", "message"],
        )
        if contact_path.exists():
            row.to_csv(contact_path, mode="a", header=False, index=False)
        else:
            row.to_csv(contact_path, index=False)

    with st.form("contact_form"):
        
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("お名前")
            email = st.text_input("メールアドレス")

        with col2:
            age = st.number_input("年齢（任意）", min_value=0, max_value=120)
            gender = st.selectbox("性別（任意）", ["選択してください", "男性", "女性", "その他"])
        
        # name = st.text_input("お名前（任意）")
        # email = st.text_input("メールアドレス（任意）")
        message = st.text_area("メッセージ")
        submitted = st.form_submit_button("送信")

        if submitted:
            if not message.strip():
                st.error("メッセージを入力してください。")
            else:
                save_contact(name, email, message)
                st.success("メッセージを送信しました。")

st.markdown(
    """
<style>
/* 本文がフッターに隠れないように下に余白を足す */
.main .block-container {
    padding-bottom: 4rem;
}

/* 画面下に張り付くフッター */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;               /* ここで画面いっぱいにする */
    background-color: #010814;  /* 本体より少し暗め */
    text-align: center;
    color: #999999;
    font-size: 0.8rem;
    padding: 0.6rem 0;
    z-index: 1000;              /* グラフより手前に出す */
}
</style>

<div class="footer">
  © 2025 nakamin
</div>
""",
    unsafe_allow_html=True,
)
