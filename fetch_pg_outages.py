from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import StringIO
import time

import pandas as pd
from playwright.sync_api import sync_playwright

TEPCO_URL = "https://teideninfo.tepco.co.jp/day/teiden/index-j.html"
OUT_DIR = Path("data/cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_COLUMNS = [
    "発生・復旧日時",
    "都県名",
    "市区町村名",
    "地区",
    "停電軒数",
    "停電理由",
    "更新日時",
    "対象日",
    "取得日時",
]

def fetch_pg_outages_past_week() -> pd.DataFrame:
    """
    Playwrightを使ってTEPCOのページから、今日を含む過去7日間（合計8日分）の停電履歴を取得。
    すべてのデータを1つのDataFrameに結合して返す。
    """
    all_dfs = []
    
    # 東京タイムゾーン
    tz = ZoneInfo("Asia/Tokyo")
    today = datetime.now(tz)
    
    date_str_list = [(today - timedelta(days=i)).strftime("%Y年%m月%d日") for i in range(8)]
    
    print(f"[INFO] 取得対象の期間（今日を含む8日間）: {date_str_list[-1]} ~ {date_str_list[0]}")

    with sync_playwright() as p:
        # headless=True で実行。もし挙動を目で確認したい場合は False にする
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        # ページを開く
        page.goto(TEPCO_URL, timeout=60000)
        
        # 画面の最初のドロップダウン（select）要素が読み込まれるのを待つ
        page.wait_for_selector("select", timeout=15000)
        
        for date_str in date_str_list:
            print(f"[FETCH] {date_str} のデータを取得中...")
            
            try:
                # 1. 最初のセレクトボックスで日付を選択（label指定）
                # TEPCOのページは日付を変更した瞬間に自動でJavaScriptの通信が走る（オンチェンジイベント）ケースがあります
                page.locator("select").first.select_option(label=date_str)
                
                # 2. 【エラー対策】ボタンクリックを廃止し、データが切り替わるのを「待つ」手法へ変更
                # 日付変更に伴う通信（ネットワークが静かになる状態）を最大5秒待つ
                try:
                    page.wait_for_load_state("networkidle", timeout=3000)
                except Exception:
                    pass # タイムアウトしても処理を進める
                
                time.sleep(1.5) # レンダリングの安全マージン
                
                # 3. HTMLを取得してPandasで解析
                html_content = page.content()
                tables = pd.read_html(StringIO(html_content))
                
                if not tables:
                    continue
                
                # ページ内で最も「列数が多い」表（＝停電情報のデータグリッド）を抽出
                df_day = max(tables, key=lambda x: x.shape[1]).copy()
                df_day.columns = [str(c).strip() for c in df_day.columns]
                
                # 「履歴情報はありません」などの文言しかない場合はスキップ
                if df_day.empty or (len(df_day) == 1 and "ありません" in str(df_day.iloc[0])):
                    print(f"-> {date_str} は停電履歴なし")
                    continue
                
                # どの日付のデータか判別できるように列を追加
                df_day["target_date"] = date_str
                all_dfs.append(df_day)
                print(f"-> {date_str} 成功: {len(df_day)} 件のレコード")
                
            except Exception as e:
                print(f"[WARN] {date_str} の取得中にエラーが発生しました: {e}")
                continue

        browser.close()

    # --- データの結合 (concat) ---
    if not all_dfs:
        print("[INFO] 対象期間内に対象となる停電データは1件もありませんでした。")
        return pd.DataFrame()
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df["fetched_at"] = today.isoformat()
    
    return combined_df

def clean_date(df):
    # データのコピーを作成
    df_clean = df.copy()

    # ==========================================================
    # 1. 「発生・復旧日時」を「発生日時」と「復旧日時」の2列に分割する
    # ==========================================================
    # スペースで分割すると、[発生日付, 発生時間, 復旧日付, 復旧時間] の4つに分かれるので、2つずつ結合します
    split_dt = df_clean["発生・復旧日時"].str.split(" ", expand=True)

    if split_dt.shape[1] == 4:
        df_clean["発生日時"] = split_dt[0] + " " + split_dt[1]
        df_clean["復旧日時"] = split_dt[2] + " " + split_dt[3]
    else:
        # 万が一、すでに復旧していてデータが特殊な場合のフォールバック
        df_clean["発生日時"] = df_clean["発生・復旧日時"]
        df_clean["復旧日時"] = "未復旧"

    # 元の結合されていた列は削除
    df_clean.drop(columns=["発生・復旧日時"], inplace=True)


    # ==========================================================
    # 2. すべての日時・日付カラムを「2026/06/20 09:29」表記に統一する
    # ==========================================================

    # 2-1. 【日時】フォーマットの統一（分まで表示）
    datetime_cols = ["発生日時", "復旧日時", "更新日時", "取得日時"]
    for col in datetime_cols:
        # 一度pandasの日時型に変換（エラーはNaTにする）
        dt_series = pd.to_datetime(df_clean[col], errors="coerce")
        # 「2026/06/20 09:29」の文字列に変換
        df_clean[col] = dt_series.dt.strftime("%Y/%m/%d %H:%M")

    # 2-2. 【日付】フォーマットの統一（年月日のみ）
    # 「2026年06月20日」を「2026/06/20」に変換
    df_clean["対象日"] = pd.to_datetime(
        df_clean["対象日"].str.replace("年", "/").str.replace("月", "/").str.replace("日", ""),
        errors="coerce"
    ).dt.strftime("%Y/%m/%d")
    
    return df

def fetch_pg_outages() -> None:
    parquet_path = OUT_DIR / "pg_outages_past_week.parquet"
    try:
        df = fetch_pg_outages_past_week()
        
        if not df.empty:
            print("---")
            print(f"総レコード数: {len(df)} 件")
            print(len(df.columns))
            df.columns = NEW_COLUMNS
            df = clean_date(df)
            print(df)
            df.to_parquet(parquet_path, index=False)
            print(f"[OK] 最新データの統合に成功し、保存しました: {parquet_path}")
        else:
            print("[INFO] 新規データが空だったため、キャッシュは更新しません。")

    except Exception as e:
        print("---")
        print("[WARN] TEPCOデータの取得中にエラーが発生しました。")
        print(f"詳細エラー: {e}")
        
        if parquet_path.exists():
            print(f"[FALLBACK] 既存のキャッシュファイルを利用します: {parquet_path}")
        else:
            print("[CRITICAL] キャッシュファイルも存在しません。初回実行時に取得を失敗した可能性があります。")

if __name__ == "__main__":
    fetch_pg_outages()