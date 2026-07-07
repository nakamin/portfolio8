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

def get_frame_with_select(page, timeout_ms: int = 30000):
    """
    page本体またはiframe内から、select要素を持つframeを探す
    Actions環境ではページ構造や読み込みタイミングがローカルと異なることがあるためpage直下だけでなくframeも確認
    """
    deadline = time.time() + timeout_ms / 1000

    while time.time() < deadline:
        # page本体 + iframe群を確認
        for frame in page.frames:
            try:
                if frame.locator("select").count() > 0:
                    return frame
            except Exception:
                continue

        time.sleep(1)

    # デバッグ用ログ
    try:
        print("[DEBUG] page url:", page.url)
        print("[DEBUG] page title:", page.title())
        print("[DEBUG] frame count:", len(page.frames))
        print("[DEBUG] html head:", page.content()[:1000])
    except Exception as e:
        print(f"[DEBUG] failed to dump page info: {e}")

    raise RuntimeError("select要素がページまたはiframe内に見つかりませんでした。")

def fetch_pg_outages_past_week() -> pd.DataFrame:
    """
    Playwrightを使って東電PGの該当ページから、今日を含む過去7日間（合計8日分）の停電履歴を取得
    - すべてのデータを1つのDataFrameに結合して返す
    """
    all_dfs = []

    tz = ZoneInfo("Asia/Tokyo")
    today = datetime.now(tz)

    date_str_list = [
        (today - timedelta(days=i)).strftime("%Y年%m月%d日")
        for i in range(8)
    ]

    print(f"[AGGREGATE] {date_str_list[-1]} ~ {date_str_list[0]}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1365, "height": 900},
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
        )

        page = context.new_page()

        page.goto(
            TEPCO_URL,
            wait_until="domcontentloaded",
            timeout=60000,
        )

        # networkidleはサイトによって終わらないことがあるので、失敗しても進める
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass

        # page直下だけでなく iframe 内も含めて select を探す
        frame = get_frame_with_select(page, timeout_ms=30000)

        select_count = frame.locator("select").count()
        print(f"[DEBUG] select_count: {select_count}")

        if select_count == 0:
            raise RuntimeError("select要素が0件です。")

        for date_str in date_str_list:
            print(f"[FETCH] Retrieving data for {date_str}")

            try:
                # 最初のselectで日付を選択
                frame.locator("select").first.select_option(label=date_str)

                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass

                time.sleep(1.5)

                # iframe内にテーブルがある場合もあるので、frame.content() を使う
                html_content = frame.content()
                tables = pd.read_html(StringIO(html_content))

                if not tables:
                    print(f"-> No tables found on {date_str}")
                    continue

                df_day = max(tables, key=lambda x: x.shape[1]).copy()
                df_day.columns = [str(c).strip() for c in df_day.columns]

                if df_day.empty or (
                    len(df_day) == 1 and "ありません" in str(df_day.iloc[0])
                ):
                    print(f"-> There was no record of power outages on {date_str}")
                    continue

                df_day["target_date"] = date_str
                all_dfs.append(df_day)

                print(f"-> {date_str} [SUCCESS] {len(df_day)} records")

            except Exception as e:
                print(f"[WARN] {date_str} : {e}")
                continue

        browser.close()

    if not all_dfs:
        print("[INFO] There were no relevant power outage data entries within the specified period")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df["fetched_at"] = today.isoformat()

    return combined_df

def clean_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    東電PG停電履歴の日時列を整形

    表示用：
    - 発生日時
    - 復旧日時
    - 更新日時
    - 取得日時
    - 対象日

    filter用：
    - 発生日時_dt
    - 復旧日時_dt
    - 更新日時_dt
    - 取得日時_dt
    - 対象日_dt
    """
    df_clean = df.copy()

    # 「発生・復旧日時」を「発生日時」と「復旧日時」に分割
    split_dt = df_clean["発生・復旧日時"].astype(str).str.split(" ", expand=True)

    if split_dt.shape[1] >= 4:
        df_clean["発生日時"] = split_dt[0] + " " + split_dt[1]
        df_clean["復旧日時"] = split_dt[2] + " " + split_dt[3]
    else:
        # 復旧日時がないケース
        df_clean["発生日時"] = df_clean["発生・復旧日時"]
        df_clean["復旧日時"] = pd.NA

    # 元の結合列は削除
    df_clean.drop(columns=["発生・復旧日時"], inplace=True)

    # 対象日を日付として解釈できる形へ
    if "対象日" in df_clean.columns:
        target_date_text = (
            df_clean["対象日"]
            .astype(str)
            .str.replace("年", "/", regex=False)
            .str.replace("月", "/", regex=False)
            .str.replace("日", "", regex=False)
        )
        df_clean["対象日_dt"] = pd.to_datetime(target_date_text, errors="coerce")
        df_clean["対象日"] = df_clean["対象日_dt"].dt.strftime("%Y/%m/%d")

    # 日時列を dt と 表示用文字列 の両方で持つ
    for col in ["発生日時", "復旧日時", "更新日時", "取得日時"]:
        if col in df_clean.columns:
            dt_col = f"{col}_dt"
            df_clean[dt_col] = pd.to_datetime(df_clean[col], errors="coerce")
            df_clean[col] = df_clean[dt_col].dt.strftime("%Y/%m/%d %H:%M")

    # 復旧日時が取れないものは表示上「未復旧」にする
    if "復旧日時" in df_clean.columns:
        df_clean["復旧日時"] = df_clean["復旧日時"].fillna("未復旧")

    return df_clean


def extract_houses(val) -> int:
    """
    '約1,030軒' や '約20軒' から数値を取り出す。
    数字がなければ0。
    """
    if pd.isna(val):
        return 0

    nums = re.findall(r"\d+", str(val).replace(",", ""))
    return int(nums[0]) if nums else 0

def filter_pg_by_target_period(
    df: pd.DataFrame,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    東電PG停電履歴を、ダッシュボード表示用の集計期間に絞る

    方針：
    - 発生日時が集計期間内の停電を残す
    - 復旧日時がある場合は、停電期間が集計期間と重なるものを残す
    - 復旧日時がない場合は、発生日時または対象日が集計期間内のものを残す
    """
    df = df.copy()

    target_start = pd.Timestamp(target_start).tz_localize(None)
    target_end = pd.Timestamp(target_end).tz_localize(None)

    # 念のため、dt列がなければ作る
    if "発生日時_dt" not in df.columns and "発生日時" in df.columns:
        df["発生日時_dt"] = pd.to_datetime(df["発生日時"], errors="coerce")

    if "復旧日時_dt" not in df.columns and "復旧日時" in df.columns:
        df["復旧日時_dt"] = pd.to_datetime(df["復旧日時"], errors="coerce")

    if "対象日_dt" not in df.columns and "対象日" in df.columns:
        df["対象日_dt"] = pd.to_datetime(df["対象日"], errors="coerce")

    # 1. 復旧日時がある停電
    # 停電期間 [発生日時, 復旧日時] が、表示期間 [target_start, target_end] と重なるもの
    has_recovery = df["復旧日時_dt"].notna()

    starts_before_end = (
        df["発生日時_dt"].isna()
        | (df["発生日時_dt"] <= target_end)
    )

    ends_after_start = (
        df["復旧日時_dt"].notna()
        & (df["復旧日時_dt"] >= target_start)
    )

    overlap_with_known_recovery = (
        has_recovery
        & starts_before_end
        & ends_after_start
    )

    # 2. 復旧日時がない停電
    # 未復旧または復旧時刻が読めないものは、発生日時または対象日が期間内なら残す
    no_recovery = df["復旧日時_dt"].isna()

    occurred_in_period = (
        df["発生日時_dt"].notna()
        & (df["発生日時_dt"] >= target_start)
        & (df["発生日時_dt"] <= target_end)
    )

    target_date_in_period = (
        df["対象日_dt"].notna()
        & (df["対象日_dt"] >= target_start.normalize())
        & (df["対象日_dt"] <= target_end.normalize())
    )

    unknown_recovery_in_period = (
        no_recovery
        & (occurred_in_period | target_date_in_period)
    )

    is_display_target = overlap_with_known_recovery | unknown_recovery_in_period

    out = df[is_display_target].copy()

    # 停電軒数を数値化して、影響が大きい順に並べやすくする
    if "停電軒数" in out.columns:
        out["停電軒数_num"] = out["停電軒数"].apply(extract_houses)
    else:
        out["停電軒数_num"] = 0

    # 原因調査中を上に出しやすくする
    out["表示優先度"] = 0

    if "停電理由" in out.columns:
        out.loc[
            out["停電理由"]
            .astype(str)
            .str.contains("調査中|確認中|原因確認中", regex=True, na=False),
            "表示優先度",
        ] += 2

    # 未復旧も優先
    if "復旧日時_dt" in out.columns:
        out.loc[out["復旧日時_dt"].isna(), "表示優先度"] += 1

    out = out.sort_values(
        ["表示優先度", "停電軒数_num", "発生日時_dt"],
        ascending=[False, False, False],
    )

    return out.reset_index(drop=True)

def summarize_tepco_teiden(df: pd.DataFrame) -> dict:
    summary = {
        "source": "東京電力 停電情報",
        "source_url": TEPCO_URL,
        "status": "success",
        "n_records": int(len(df)),
        "total_affected_houses": 0,
        "under_investigation_count": 0,
        "max_affected_houses": 0,
        "max_affected_area": None,
    }

    if df.empty:
        return summary

    df = df.copy()

    if "停電軒数" in df.columns:
        df["停電軒数_num"] = df["停電軒数"].apply(extract_houses)
        summary["total_affected_houses"] = int(df["停電軒数_num"].sum())
        summary["max_affected_houses"] = int(df["停電軒数_num"].max())

        if summary["max_affected_houses"] > 0:
            max_idx = df["停電軒数_num"].idxmax()
            max_row = df.loc[max_idx]

            area_parts = []
            for col in ["都県名", "市区町村名", "地区"]:
                if col in df.columns and pd.notna(max_row.get(col)):
                    area_parts.append(str(max_row.get(col)))

            summary["max_affected_area"] = " ".join(area_parts) if area_parts else None

    if "停電理由" in df.columns:
        summary["under_investigation_count"] = int(
            df["停電理由"]
            .astype(str)
            .str.contains("調査中|確認中|原因確認中", regex=True, na=False)
            .sum()
        )

    return summary

def fetch_pg_outages() -> None:
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    display_now = now.strftime("%Y/%m/%d %H:%M")

    # PG側は「過去7日〜今日」までを集計対象にする
    past_start = pd.to_datetime(now - timedelta(days=7)).normalize()
    target_end = (
        pd.to_datetime(now).normalize()
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )

    print(f"[AGGREGATE] {past_start} ~ {target_end}")

    try:
        df = fetch_pg_outages_past_week()

        if not df.empty:
            print("---")
            print(f"Total before filter: {len(df)} records")

            # カラム名を上書きして整形
            df.columns = NEW_COLUMNS

            # 日時整形
            df = clean_date(df)

            # 集計期間に合わせてfilter
            df = filter_pg_by_target_period(df, past_start, target_end)

            print(f"[FILTER] valid time: {len(df)} records")
            print(df)

            summary = summarize_tepco_teiden(df)
            summary["updated_at"] = display_now
            summary["target_start"] = past_start.strftime("%Y/%m/%d")
            summary["target_end"] = target_end.strftime("%Y/%m/%d")
            summary["status"] = "success"

            # 成功したらParquetとJSONを更新
            df.to_parquet(PARQUET_PATH, index=False)

            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"[OK] The latest data has been successfully integrated and saved: {PARQUET_PATH}")
            print(summary)

        else:
            print("[INFO] No power outage data entries were found in fetched period.")


            empty_df = pd.DataFrame(columns=[
                "都県名",
                "市区町村名",
                "地区",
                "停電軒数",
                "停電理由",
                "発生日時",
                "復旧日時",
                "更新日時",
                "対象日",
                "取得日時",
            ])

            summary = summarize_tepco_teiden(empty_df)
            summary["updated_at"] = display_now
            summary["target_start"] = past_start.strftime("%Y/%m/%d")
            summary["target_end"] = target_end.strftime("%Y/%m/%d")
            summary["status"] = "success"

            empty_df.to_parquet(PARQUET_PATH, index=False)

            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print("[OK] Empty TEPCO outage cache saved.")
            print(summary)

    except Exception as e:
        print("---")
        print("[WARN] An error occurred while acquiring power outage data")
        print(f"details: {e}")

        if PARQUET_PATH.exists():
            print(f"[FALLBACK] Found cache file: {PARQUET_PATH}. Regenerating summary from cache.")

            try:
                df_cached = pd.read_parquet(PARQUET_PATH)
                summary = summarize_tepco_teiden(df_cached)

                summary["updated_at"] = display_now
                summary["target_start"] = past_start.strftime("%Y/%m/%d")
                summary["target_end"] = target_end.strftime("%Y/%m/%d")
                summary["status"] = "fallback_cache"
                summary["error_log"] = str(e)

                with open(JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

                print("[OK] TEPCO power outage summary updated using existing cache data.")
                print(summary)

            except Exception as cache_err:
                print(f"[CRITICAL] Failed to read from cache file: {cache_err}")

        else:
            print("[CRITICAL] Cache file does not exist. Creating empty/failed summary.")

            error_summary = {
                "source": "東京電力 停電情報",
                "source_url": TEPCO_URL,
                "updated_at": display_now,
                "target_start": past_start.strftime("%Y/%m/%d"),
                "target_end": target_end.strftime("%Y/%m/%d"),
                "status": "failed",
                "error": str(e),
                "n_records": 0,
                "total_affected_houses": 0,
                "under_investigation_count": 0,
            }

            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(error_summary, f, ensure_ascii=False, indent=2)

    print("[INFO] TEPCO power outage processing step passed.")

if __name__ == "__main__":
    fetch_pg_outages()