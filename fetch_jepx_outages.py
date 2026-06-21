from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import pytz
import pandas as pd
from playwright.sync_api import sync_playwright


JEPX_OUTAGES_URL = "https://hjks.jepx.or.jp/hjks/outages"
OUT_DIR = Path("data/cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)
JST = pytz.timezone("Asia/Tokyo")

PARQUET_OATH = OUT_DIR / "jepx_outages_latest.parquet"
JSON_PATH = OUT_DIR / "jepx_outages_summary.json"

JEPX_COLUMNS = [
    "エリア",
    "発電事業者",
    "発電所コード",
    "発電所名",
    "発電形式",
    "ユニット名",
    "認可出力",
    "停止区分",
    "種別",
    "低下量",
    "停止日時",
    "復旧見通し",
    "復旧予定日",
    "停止原因",
    "最終更新日時",
]

def _to_numeric_kw(series: pd.Series) -> pd.Series:
    """
    '1,000' や '-' などが混ざる列を数値に変換するための関数（kW単位の列を想定）
    """
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("－", "", regex=False)
        .str.replace("-", "", regex=False)
        .replace("", pd.NA)
        .pipe(pd.to_numeric, errors="coerce")
    )

def add_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "認可出力" in df.columns:
        df["認可出力_kW"] = _to_numeric_kw(df["認可出力"])
        df["認可出力_MW"] = df["認可出力_kW"] / 1000

    if "低下量" in df.columns:
        df["低下量_kW"] = _to_numeric_kw(df["低下量"])
        df["低下量_MW"] = df["低下量_kW"] / 1000

    # 停止の場合、低下量が空欄で、認可出力だけ入っている→「影響量」は、低下量があれば低下量、なければ認可出力を使う
    if "低下量_kW" in df.columns and "認可出力_kW" in df.columns:
        df["影響量_kW"] = df["低下量_kW"].fillna(df["認可出力_kW"])
        df["影響量_MW"] = df["影響量_kW"] / 1000

    return df

def read_jepx_csv(download_path: Path) -> pd.DataFrame:
    """
    JEPX停止情報CSVを読み込む
    """
    encodings = ["cp932", "utf-8-sig", "utf-8"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(
                download_path,
                encoding=enc,
                header=0,
                on_bad_lines="skip",
            )
            break
        except Exception as e:
            last_error = e
    else:
        raise RuntimeError(f"CSVの読み込みに失敗しました: {last_error}")
    
    df = df.reset_index(drop=False)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = JEPX_COLUMNS

    return df

def fetch_jepx_outages_tocsv() -> pd.DataFrame:
    """
    Playwrightを使ってブラウザを起動し、CSVダウンロードボタンをクリックしてデータを取得する
    """
    
    with sync_playwright() as p:
        # headless=True でバックグラウンド実行（挙動を見たい時は False にする）
        browser = p.chromium.launch(headless=True)
        
        # User-Agentを一般のブラウザに偽装してセキュリティブロックを回避
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        # ページへ移動
        page.goto(JEPX_OUTAGES_URL, timeout=60000)
        
        page.locator("select").first.select_option(label="東京")
        page.get_by_role("button", name="検索").click()
        
        # --- CSVダウンロードのトリガー ---
        with page.expect_download() as download_info:
            page.get_by_role("button", name="CSVダウンロード").click()

        download = download_info.value

        # Playwrightが内部的に保存した一時ファイルのパス
        downloaded_path = Path(download.path())

        # browser/contextを閉じる前に読み込む
        df = read_jepx_csv(downloaded_path)
        df = add_numeric_columns(df)
        browser.close()
    return df

def summarize_jepx(df: pd.DataFrame) -> dict:
    now = datetime.now(ZoneInfo("Asia/Tokyo"))

    summary = {
        "source": "JEPX 発電情報公開システム 停止情報一覧",
        "source_url": JEPX_OUTAGES_URL,
        "updated_at": now.isoformat(),
        "status": "success",
        "n_records": int(len(df)),
        "total_decrease_mw": None,
        "total_affected_mw": None,
        "unplanned_count": None,
        "unknown_recovery_count": None,
    }

    if "低下量_MW" in df.columns:
        summary["total_decrease_mw"] = float(df["低下量_MW"].fillna(0).sum())

    if "影響量_MW" in df.columns:
        summary["total_affected_mw"] = float(df["影響量_MW"].fillna(0).sum())

    if "停止区分" in df.columns:
        summary["unplanned_count"] = int(
            df["停止区分"].astype(str).str.contains("計画外", na=False).sum()
        )

    if "復旧見通し" in df.columns:
        summary["unknown_recovery_count"] = int(
            df["復旧見通し"]
            .astype(str)
            .str.contains("未定|なし|不明", regex=True, na=False)
            .sum()
        )

    return summary

def filter_jepx_by_target_period(
    df: pd.DataFrame,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> pd.DataFrame:
    
    df = df.copy()

    df["停止日時_dt"] = pd.to_datetime(df["停止日時"], errors="coerce")
    df["復旧予定日_dt"] = pd.to_datetime(df["復旧予定日"], errors="coerce")

    # 復旧予定日が日付だけの場合、その日の終わりまで影響するとみなす
    has_recovery = df["復旧予定日_dt"].notna()
    df.loc[has_recovery, "復旧予定日_dt"] = (
        df.loc[has_recovery, "復旧予定日_dt"]
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )

    target_start = pd.Timestamp(target_start).tz_localize(None)
    target_end = pd.Timestamp(target_start).tz_localize(None)
    # 1. 停止の始まりが表示期間の終わりより前（または開始不明）
    starts_before_end = df["停止日時_dt"].isna() | (df["停止日時_dt"] <= target_end)
    # 2. 復旧が表示期間の始まりより後（またはまだ復旧していない/未定）
    ends_after_start = df["復旧予定日_dt"].isna() | (df["復旧予定日_dt"] >= target_start)

    # この2つが両方成り立つものが「期間中に1日でも止まっていた」データ
    is_overlapping = starts_before_end & ends_after_start
    return df[is_overlapping].copy()

def fetch_jepx_outages() -> None:
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    display_now = now.strftime("%Y/%m/%d %H:%M")
    past_start = pd.to_datetime(now - timedelta(days=7)).normalize()
    pred_end = pd.to_datetime(now + timedelta(days=6)).normalize()
    print(f"[AGGREGATE] {past_start} ~ {pred_end}")

    try:
        df = fetch_jepx_outages_tocsv()
        print("[SUCCESS] download JEPX outage data")
        df = filter_jepx_by_target_period(df, past_start, pred_end)
        print("[FILTER] valid time: \n", df)
        summary = summarize_jepx(df)
        summary["updated_at"] = display_now

        # 成功したらParquetとJSONを更新（キャッシュ更新）
        df.to_parquet(PARQUET_OATH, index=False)
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("[OK] JEPX outage data updated with fresh web data.")
        print(summary)

    except Exception as e:
        
        if PARQUET_OATH.exists():
            print(f"[FALLBACK] Found cache file: {PARQUET_OATH}. Regenerating summary from cache.")
            try:
                # キャッシュからデータを読み込み、サマリーを再作成
                df_cached = pd.read_parquet(PARQUET_OATH)
                summary = summarize_jepx(df_cached)
                
                # エラーが起きたが、データ自体はキャッシュで維持できている状態を記録
                summary["updated_at"] = display_now
                summary["status"] = "fallback_cache"
                summary["error_log"] = str(e) # デバッグ用にエラー内容も残しておく
                
                with open(JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                    
                print("[OK] JEPX summary updated using existing cache data.")
                print(summary)
                
            except Exception as cache_err:
                print(f"[CRITICAL] Failed to read from cache file: {cache_err}")
        else:
            # キャッシュすら存在しない場合
            print("[CRITICAL] Cache file does not exist. Creating empty/failed summary.")
            error_summary = {
                "source": "JEPX 発電情報公開システム 停止情報一覧",
                "source_url": JEPX_OUTAGES_URL,
                "updated_at": display_now,
                "status": "failed",
                "error": str(e),
                "n_records": 0
            }
            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(error_summary, f, ensure_ascii=False, indent=2)

    print("[INFO] JEPX processing step passed.")

if __name__ == "__main__":
    fetch_jepx_outages()