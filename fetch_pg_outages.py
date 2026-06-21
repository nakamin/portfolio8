from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import StringIO
import time
import re, json

import pandas as pd
from playwright.sync_api import sync_playwright

TEPCO_URL = "https://teideninfo.tepco.co.jp/day/teiden/index-j.html"
OUT_DIR = Path("data/cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = OUT_DIR / "pg_outages_past_week.parquet"
JSON_PATH = OUT_DIR / "pg_outages_summary.json"

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
    Playwrightを使って東電PGの該当ページから、今日を含む過去7日間（合計8日分）の停電履歴を取得
    - すべてのデータを1つのDataFrameに結合して返す
    """
    all_dfs = []
    
    # 東京タイムゾーン
    tz = ZoneInfo("Asia/Tokyo")
    today = datetime.now(tz)
    
    date_str_list = [(today - timedelta(days=i)).strftime("%Y年%m月%d日") for i in range(8)]
    
    print(f"[AGGREGATE] {date_str_list[-1]} ~ {date_str_list[0]}")

    with sync_playwright() as p:

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
            print(f"[FETCH] Retrieving data for {date_str}")
            
            try:
                # 最初のセレクトボックスで日付を選択（label指定）
                page.locator("select").first.select_option(label=date_str)
                
                try:
                    page.wait_for_load_state("networkidle", timeout=3000)
                except Exception:
                    pass # タイムアウトしても処理を進める
                
                time.sleep(1.5) # レンダリングの安全マージン
                
                # HTMLを取得してPandasで解析
                html_content = page.content()
                tables = pd.read_html(StringIO(html_content))
                
                if not tables:
                    continue
                
                # ページ内で最も「列数が多い」表（＝停電情報のデータグリッド）を抽出
                df_day = max(tables, key=lambda x: x.shape[1]).copy()
                df_day.columns = [str(c).strip() for c in df_day.columns]
                
                # 「履歴情報はありません」などの文言しかない場合はスキップ
                if df_day.empty or (len(df_day) == 1 and "ありません" in str(df_day.iloc[0])):
                    print(f"-> There was no record of power outages on {date_str}")
                    continue
                
                # どの日付のデータか判別できるように列を追加
                df_day["target_date"] = date_str
                all_dfs.append(df_day)
                print(f"-> {date_str} [SUCCESS] {len(df_day)} records")
                
            except Exception as e:
                print(f"[WARN] {date_str} : {e}")
                continue

        browser.close()

    # --- データの結合 (concat) ---
    if not all_dfs:
        print("[INFO] There were no relevant power outage data entries within the specified period")
        return pd.DataFrame()
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df["fetched_at"] = today.isoformat()
    
    return combined_df

def clean_date(df):
    # データのコピーを作成
    df_clean = df.copy()

    # 「発生・復旧日時」を「発生日時」と「復旧日時」の2列に分割する
    split_dt = df_clean["発生・復旧日時"].str.split(" ", expand=True)

    if split_dt.shape[1] == 4:
        df_clean["発生日時"] = split_dt[0] + " " + split_dt[1]
        df_clean["復旧日時"] = split_dt[2] + " " + split_dt[3]
    else:
        df_clean["発生日時"] = df_clean["発生・復旧日時"]
        df_clean["復旧日時"] = "未復旧"

    # 元の結合されていた列は削除
    df_clean.drop(columns=["発生・復旧日時"], inplace=True)

    # すべての日時・日付カラムを「2026/06/20 09:29」表記に統一する
    # 【日時】フォーマットの統一（分まで表示）
    datetime_cols = ["発生日時", "復旧日時", "更新日時", "取得日時"]
    for col in datetime_cols:
        # 一度pandasの日時型に変換（エラーはNaTにする）
        dt_series = pd.to_datetime(df_clean[col], errors="coerce")
        # 「2026/06/20 09:29」の文字列に変換
        df_clean[col] = dt_series.dt.strftime("%Y/%m/%d %H:%M")

    # 【日付】フォーマットの統一（年月日のみ）
    # 「2026年06月20日」を「2026/06/20」に変換
    df_clean["対象日"] = pd.to_datetime(
        df_clean["対象日"].str.replace("年", "/").str.replace("月", "/").str.replace("日", ""),
        errors="coerce"
    ).dt.strftime("%Y/%m/%d")
    
    return df_clean

def summarize_tepco_teiden(df: pd.DataFrame) -> dict:
    now = datetime.now(ZoneInfo("Asia/Tokyo"))

    summary = {
        "source": "東京電力 停電情報",
        "source_url": TEPCO_URL,
        "updated_at": now.isoformat(),
        "status": "success",
        "n_records": int(len(df)),
        "total_affected_houses": 0,    # 合計停電軒数（約120軒などを数値化）
        "under_investigation_count": 0, # 原因調査中の件数
    }

    # 合計停電軒数の集計（"約1,030軒" や "約20軒" から数値を抽出して合計）
    if "停電軒数" in df.columns:
        def extract_houses(val):
            if pd.isna(val):
                return 0
            # 文字列から数字（カンマ含む）だけを抽出
            nums = re.findall(r'\d+', str(val).replace(',', ''))
            return int(nums[0]) if nums else 0

        summary["total_affected_houses"] = int(
            df["停電軒数"].apply(extract_houses).sum()
        )

    # 原因調査中の件数カウント
    if "停電理由" in df.columns:
        summary["under_investigation_count"] = int(
            df["停電理由"].astype(str).str.contains("調査中", na=False).sum()
        )

    return summary

def fetch_pg_outages() -> None:
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    display_now = now.strftime("%Y/%m/%d %H:%M")

    try:
        df = fetch_pg_outages_past_week()
        
        if not df.empty:
            print("---")
            print(f"Total: {len(df)} records")
            
            # カラム名を上書きして整形
            df.columns = NEW_COLUMNS
            df = clean_date(df)
            print(df)
            summary = summarize_tepco_teiden(df)
            summary["updated_at"] = display_now

            # 成功したらParquetとJSONを更新（キャッシュ更新）
            df.to_parquet(PARQUET_PATH, index=False)
            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"[OK] The latest data has been successfully integrated and saved: {PARQUET_PATH}")
            print(summary)
        else:
            print("[INFO] The cache will not be updated because there is no new data")

    except Exception as e:
        print("---")
        print("[WARN] An error occurred while acquiring power outage data")
        print(f"details: {e}")
        
        if PARQUET_PATH.exists():
            print(f"[FALLBACK] Found cache file: {PARQUET_PATH}. Regenerating summary from cache.")
            try:
                # キャッシュからデータを読み込み、サマリーを再作成
                df_cached = pd.read_parquet(PARQUET_PATH)
                summary = summarize_tepco_teiden(df_cached)
                
                # エラーが起きたが、データ自体はキャッシュで維持できている状態を記録
                summary["updated_at"] = display_now
                summary["status"] = "fallback_cache"
                summary["error_log"] = str(e)  # デバッグ用にエラー内容を記録
                
                with open(JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                    
                print("[OK] TEPCO power outage summary updated using existing cache data.")
                print(summary)
                
            except Exception as cache_err:
                print(f"[CRITICAL] Failed to read from cache file: {cache_err}")
        else:
            # キャッシュすら存在しない場合
            print("[CRITICAL] Cache file does not exist. Creating empty/failed summary.")
            error_summary = {
                "source": "東京電力 停電情報",
                "source_url": TEPCO_URL,
                "updated_at": display_now,
                "status": "failed",
                "error": str(e),
                "n_records": 0
            }
            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(error_summary, f, ensure_ascii=False, indent=2)

    print("[INFO] TEPCO power outage processing step passed.")

if __name__ == "__main__":
    fetch_pg_outages()