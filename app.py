from pathlib import Path
from datetime import datetime, timedelta
import json
import os, requests, io

import pandas as pd
import pytz
import streamlit as st
import traceback

from visualize.plot_energy_mix import plot_energy_mix
from visualize.plot_price import plot_price
from visualize.plot_demand import plot_demand

# ========================
# 基本設定
# ========================

CACHE_DIR = Path("data/cache")
STATIC_DIR = Path("data/static")
JST = pytz.timezone("Asia/Tokyo")

RUN_ENV = os.getenv("RUN_ENV", "local") 
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/nakamin/portfolio8/main"

HERO_GITHUB_URL = (
    "https://raw.githubusercontent.com/nakamin/portfolio8/main/data/static/hero.png"
)

st.set_page_config(
    layout="wide",
)

# ========================
# 共通ユーティリティ
# ========================
def _read_parquet_from_github(rel_path: str) -> pd.DataFrame:
    """
    - GitHub の raw URL から parquet を読み込む
    - rel_path はレポジトリ root からの相対パス
    """
    url = f"{GITHUB_RAW_BASE}/{rel_path}"
    resp = requests.get(url)
    resp.raise_for_status()
    df = pd.read_parquet(io.BytesIO(resp.content))
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _read_parquet_local(path: Path) -> pd.DataFrame:
    """
    - ローカル or Actions で、ファイルシステム上の parquet を読む
    """
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data(show_spinner=False)
def show_hero():
    _, col, _ = st.columns([1, 20, 1])  # 真ん中のカラムを広めに
    with col:
        if RUN_ENV == "hf":
            # HF では GitHub の画像を直接表示
            st.image(HERO_GITHUB_URL)
        else:
            hero_path = STATIC_DIR / "hero.png"
            if hero_path.exists():
                st.image(str(hero_path))

@st.cache_data(show_spinner=False)
def load_parquet(name: str, version_key: str | None = None) -> pd.DataFrame | None:
    """
    name: 'demand_forecast' のようなベース名
    - local: data/cache/{name}.parquet
    - hf:    GitHub raw の data/cache/{name}.parquet
    - version_key: これが変わるとキャッシュが無効化されて再読込される
    """
    try:
        if RUN_ENV == "hf":
            rel_path = f"data/cache/{name}.parquet"
            df = _read_parquet_from_github(rel_path)
        else:
            path = CACHE_DIR / f"{name}.parquet"
            df = _read_parquet_local(path)
        return df
    except Exception as e:
        st.error(f"{name}.parquet を読み込めませんでした: {e}")
        return None

def load_metadata() -> dict:
    """
    メタ情報（sources, last_updated など）を取得
    - local 環境: data/cache/metadata.json を読む
    - hf 環境: GitHub raw の data/cache/metadata.json を読む
    """
    if RUN_ENV == "hf":
        rel_path = "data/cache/metadata.json"
        url = f"{GITHUB_RAW_BASE}/{rel_path}"
        resp = requests.get(url)
        if resp.status_code != 200:
            # 初回など、まだ metadata.json が無いとき
            return {}
        try:
            return resp.json()
        except ValueError:
            # 念のため、JSONとして読めないときは空にする
            return {}
    else:
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

def main():
    
    # st.caption(f"RUN_ENV = {RUN_ENV}")

    # meta = load_metadata()
    # st.caption(f"meta keys = {list(meta.keys())}")
    # st.caption(f"last_updated_jst = {meta.get('last_updated_jst')}")
    # version = meta.get("last_updated_jst", "no-meta")
    # st.caption(f"version_key = {version}")
    
    show_hero()

    now_jst = datetime.now(JST)
    st.caption(f"現在時刻（JST）: {now_jst.strftime('%Y-%m-%d %H:%M')}")

    meta = load_metadata()
    if meta.get("last_updated_jst"):
        last_jst = datetime.fromisoformat(meta["last_updated_jst"])
        st.caption(f"データ更新時刻: {last_jst.strftime('%Y-%m-%d %H:%M')}（自動更新日時）")
    else:
        st.caption("データ更新時刻: 不明")
    
    version = meta.get("last_updated_jst", "no-meta")

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

        demand = load_parquet("demand_forecast", version_key=version)      # predicted_demand / realized_demand / demand など
        if demand is None:
            st.stop()
        print("demand: \n", demand)

        fig_d = plot_demand(demand, now_floor)
        st.plotly_chart(fig_d)
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

        price_fc = load_parquet("price_forecast", version_key=version)          # price / predicted_price(10/50/90) / tokyo_price_jpy_per_kwh
        if price_fc is None:
            st.stop()
        print("price_fc: \n", price_fc)

        fig_p = plot_price(price_fc=price_fc, today=today, now_floor=now_floor)
        st.plotly_chart(fig_p)
        
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

        dispatch = load_parquet("dispatch_optimal", version_key=version)
        if dispatch is None:
            st.stop()
        print("dispatch: \n", dispatch)

        fig_balance = plot_energy_mix(dispatch, now_floor)
        st.plotly_chart(fig_balance)
        
        st.markdown(
        f"""
        <h4 style="text-align:center; margin-top: 1rem;">
        予測対象日：{today:%Y年%m月%d日}
        </h4>
        """,
        unsafe_allow_html=True,
    )


    with tab_model:
        meta = load_metadata()
        sources = meta.get("sources", {})

        st.markdown("## モデル一覧（需要 → 価格 → 最適化）")
        st.caption(
            "このダッシュボードは「需要予測 → 価格予測 → 電源構成の最適化」の順に計算し、"
            "`data/cache/*.parquet` を更新しています。ここでは各モデルの役割・入力・出力をまとめます。"
        )

        st.markdown("---")
        st.markdown("### 需要予測モデル（Demand Forecast）")

        st.markdown(
            f"""
    **目的**  
    - 30分刻みで「明日〜7日先」までの需要（MW）を予測  

    **入力データ**  
    - 気象（実績+予報）：気温 / 湿度 など（`weather_bf1w_af1w.parquet`）  
    - カレンダー特徴：月・時刻（sin/cos 変換）、休日フラグ（祝日 + 年末年始 + お盆）  
    - 体感温度に近い補助特徴：`temperature_abs = |temperature - 18.1|`  

    **モデル**  
    - {sources.get("demand_model", "GRU（時系列モデル）")}  
    - 学習済み重み（`.pth`）と目的変数スケーラ（`scaler_y.pkl`）は Hugging Face Hub に置き、
    GitHub には毎日更新される cache のみを置く設計です。

    **出力**  
    - `demand_forecast.parquet`  
    - `timestamp`（30分刻み）  
    - `predicted_demand`（需要予測 MW）  
    - `realized_demand`（取得できる範囲のみ実績 MW）  
    - `demand`（表示用：予測があれば予測、なければ実績で埋めた列）
            """
        )

        st.markdown("---")
        st.markdown("### 価格予測モデル（Price Forecast）")

        st.markdown(
            f"""
    **目的**  
    - 東京エリアの JEPX スポット価格（円/kWh）を、将来の不確実性（リスクレンジ）込みで推定
    - ダッシュボード上では「P10 / P50 / P90」のような分位点（下振れ・中央値・上振れ）で表示する

    **入力データ**  
    - 需要：需要予測（`predicted_demand`）および必要なら実績需要  
    - 価格ラグ：過去のJEPX価格（24時間前）  
    - 市況：燃料（Brent / Henry Hub / 石炭など）・為替（USD/JPY など）、ラグ
    
    
      
    - 天候：気温、日照時間、全天日射、風速

    **モデル**  
    - {sources.get("price_model", "GAM + LightGBM（分位点回帰）")}  
    - **GAM**：季節性・時刻（sin/cos）・休日など「なめらかな周期構造」を捉える（基礎形状）  
    - **LightGBM**：燃料・為替・ラグ等の非線形関係を捉え、GAMだけでは取り切れない変動を補正  
    - 予測時は、学習時と同じ前処理（カテゴリの扱い、欠損処理、特徴量列の順序）を厳密に揃えます。

    **出力**  
    - `price_forecast.parquet`（想定）  
    - `timestamp`  
    - `predicted_price_p10`, `predicted_price_p50`, `predicted_price_p90`  
    - `tokyo_price_jpy_per_kwh`（取得できる範囲のみ実績）  
    - `price`（表示用：予測があれば予測、なければ実績で埋めた列）
            """
        )

        st.markdown("---")
        st.markdown("### 最適化モデル（Dispatch Optimization）")

        st.markdown(
            f"""
    **目的**  
    - 需要予測を満たしつつ、1日分（30分×48）の供給コストを最小化する電源構成を求める
    - ダッシュボードでは、電源別の積み上げ（PV/風力/水力/火力/蓄電池/揚水/受電…）を可視化

    **入力**  
    - 需要：需要予測（`predicted_demand` または `demand`）  
    - 再エネ上限：太陽光・風力の時刻別上限（予報や設備容量から算出）  
    - 設備制約：火力・水力などの最大/最小、ランプ制約  
    - 蓄電池・揚水：容量、充放電効率、入出力上限、SOC制約  
    - コスト：燃料単価・起動費・不足（shedding）ペナルティなど

    **モデル**  
    - {sources.get("optimizer", "数理最適化（線形/混合整数など）")}  
    - 「予測→最適化→結果保存」までを毎日自動で回し、最新の電源構成を更新します。

    **出力**  
    - `opt_dispatch.parquet`（想定）  
    - `timestamp`, `pv`, `wind`, `hydro`, `coal`, `lng`, `oil`, `battery`, `pumped`, `import`, `curtail_*`, `shed`, `total_cost` など  
            """
        )


    with tab_pipeline:
        meta = load_metadata()
        src = meta.get("sources", {})
        markets = src.get("markets", {})

        st.markdown("## パイプライン概要（毎日自動更新）")
        st.caption(
            "このプロジェクトは「毎日データを取り直し → 予測 → 最適化 → cache保存」を繰り返しています。"
        )

        st.markdown("### 全体フロー")
        st.markdown(
            """
    1. **データ取得**：需要実績 / 天気（実績+予報）/ 市場価格（JEPX）/ 燃料・為替  
    2. **前処理・特徴量生成**：モデル入力の整形（欠損処理、休日フラグ、ラグ、移動平均など）  
    3. **予測**：需要 → 価格（分位点）  
    4. **最適化**：1日分の電源構成を最適化  
    5. **保存**：`data/cache/*.parquet` と `metadata.json` を更新し、ダッシュボードが参照
            """
        )

        st.markdown("---")
        st.markdown("### 1. データ取得フェーズ（`fetch_*.py`）")

        st.markdown("**需要（実績）・価格（実績）**")
        st.markdown(
            f"""
    - 需要実績: {src.get("demand", "TEPCO でんき予報 エリア需給実績データ）")}  
    - 市場価格: {src.get("jepx", "JEPX 日前スポット（東京エリア）")}  
    - JEPXはページ操作後にダウンロードが発生するため、Playwrightでブラウザ操作を自動化しています。
            """
        )

        st.markdown("**天気（実績+予報）**")
        st.markdown(
            f"""
    - {src.get('weather', 'Open-Meteo（JMAベース）など')}  
    - 「一週間前〜一週間後」をまとめて取得し、需要・価格の特徴量に利用します。
            """
        )

        st.markdown("**燃料市況・為替**")
        st.markdown(
            "\n".join(
                [
                    f"- 為替: {markets.get('fx', 'Frankfurter / ECB など')}",
                    f"- 原油: {markets.get('oil', 'EIA / yfinance BZ=F など')}",
                    f"- LNG: {markets.get('lng', 'EIA / yfinance NG=F など')}",
                    f"- 石炭: {markets.get('coal', 'World Bank Pink Sheet など')}",
                ]
            )
        )

        st.markdown("---")
        st.markdown("### 2. 前処理・特徴量生成（`predict_demand.py`, `predict_price.py`）")

        st.markdown("**需要予測の特徴量**")
        st.markdown(
            f"""
    - 利用データ: 気象（実績+予報）、休日フラグ、月・時刻（sin/cos）、補助特徴（temperature_abs）など  
    - モデル: {src.get("demand_model", "GRU")}  
    - 出力: `demand_forecast.parquet`（予測 + 実績の統合）
            """
        )

        st.markdown("**価格予測の特徴量**")
        st.markdown(
            f"""
    - 利用データ: 需要予測、JEPX過去価格（ラグ/移動平均）、燃料・為替、天候など  
    - モデル: {src.get("price_model", "GAM + LightGBM（分位点）")}  
    - 出力: P10 / P50 / P90（またはP10/P50/P90相当）の分位点価格
            """
        )

        st.markdown("---")
        st.markdown("### 3. 最適化（`optimize_dispatch.py`）")
        st.markdown(
            f"""
    - 需要（予測）を満たす電源構成をコスト最小化で求める 
    - 入力: 需要予測、燃料コスト、再エネ上限、設備制約など  
    - モデル: {src.get("optimizer", "数理最適化モデル")}  
    - 出力: 電源別の出力とコスト（`opt_dispatch.parquet` など）
            """
        )

        st.markdown("---")
        st.markdown("### 4. キャッシュ・ダッシュボード連携（`run_daily_pipeline.py`）")
        st.markdown(
            """
    - 上記 1〜3 を一括実行し、`data/cache/*.parquet` を生成します。 
    - 併せて `metadata.json` に「最終更新時刻」「データソース」「モデル情報」を保存します。  
    - ダッシュボードは `metadata.json` を読み込み、画面上部の更新日時や「データ・モデル情報」タブ、
    フッターのクレジットに反映しています。
            """
        )


    with contact:
        st.subheader("お問い合わせフォーム")
        st.markdown("モデルや可視化に関するご意見等がございましたら下記フォームよりお問い合わせください。")

        # ★ age/gender も保存するなら、保存関数の引数とCSV列を増やす
        def save_contact(name: str, email: str, age: int, gender: str, message: str):
            contact_path = Path("data/contact_messages.csv")
            contact_path.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")

            row = pd.DataFrame(
                [[ts, name, email, age, gender, message]],
                columns=["timestamp", "name", "email", "age", "gender", "message"],
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
                age = st.number_input("年齢（任意）", min_value=0, max_value=120, value=0)
                gender = st.selectbox("性別（任意）", ["選択してください", "男性", "女性", "その他"])

            message = st.text_area("メッセージ")
            submitted = st.form_submit_button("送信")

            if submitted:
                if not message.strip():
                    st.error("メッセージを入力してください。")
                else:
                    # gender が未選択なら空扱いにして保存
                    gender_value = "" if gender == "選択してください" else gender
                    save_contact(name, email, int(age), gender_value, message)
                    st.success("メッセージを送信しました。")

    st.empty()
    st.markdown("<div style='height:100px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
    <style>
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("アプリ実行中にエラーが発生しました")
        st.code(traceback.format_exc())