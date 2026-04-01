from datetime import datetime, timedelta, timezone
import re, os, time
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

JST = timezone(timedelta(hours=9))
CACHE_DIR = Path("data/cache")
RAW_DIR = Path("data/raw")
DEBUG_DIR = Path("data/debug")

RAW_OUT_CURRENT = RAW_DIR / "spot_tokyo_current.csv"
RAW_OUT_PREV = RAW_DIR / "spot_tokyo_prev.csv"
PROC_OUT = CACHE_DIR / "spot_tokyo_bf1w_tdy.parquet"

URL = "https://www.jepx.jp/electricpower/market-data/spot/"

def today_jst_date():
    return datetime.now(JST).date()

def get_fiscal_year(date):
    """日付から年度（4月始まり）を返す"""
    return date.year if date.month >= 4 else date.year - 1

def click_today_cell_only(page):
    """
    - カレンダーを開いて『今日セル』だけをクリック
    - 年/月はデフォルトで当月になっているので触らない
    """
    # トレース & スクショ
    page.context.tracing.start(screenshots=True, snapshots=True, sources=True)

    y = today_jst_date()
    dd = y.day

    # プレースホルダ『日付を選択してください。』をクリック → パネル表示待ち
    page.get_by_text("日付を選択してください。").first.click(timeout=10_000)
    panel = page.locator(".ui-datepicker")
    panel.first.wait_for(state="visible", timeout=10_000) # 可視化を待つ

    # 当月かつ有効な日セルの a/button を XPath で取得 → クリック
    xpath = (
        ".//td[not(contains(@class,'other')) and " # 前後月や無効セルを除外
        "     not(contains(@class,'disabled'))]"
        "//*[self::a or self::button or self::div]"
        f"[normalize-space(text())='{dd}']"
    )
    day = panel.locator(f"xpath={xpath}")

    try:
        day.first.wait_for(state="visible", timeout=5_000)
        day.first.click(timeout=10_000, force=True)
    except PlaywrightTimeoutError as e:
        # デバッグ出力して落とす
        page.screenshot(path=os.path.join(DEBUG_DIR, "calendar_page.png"), full_page=True)
        try: panel.first.screenshot(path=os.path.join(DEBUG_DIR, "calendar_panel.png"))
        except: pass
        page.context.tracing.stop(os.path.join(DEBUG_DIR, "trace.zip"))
        raise RuntimeError(
            "日セルをクリックできませんでした。debug: debug/calendar_page.png / calendar_panel.png / trace.zip"
        ) from e

def download_spot_csv(page, modal, fiscal_year, save_path, is_current=True):
    """
    指定された年度を選択してダウンロードする
    今年度(is_current=True)の場合はプルダウン操作をスキップ
    """
    if not is_current:
        print(f"Switching to previous year: {fiscal_year}年度")
        # プルダウン（select要素）を探して、テキストまたは値で選択
        try:
            # 1. select要素がある場合
            modal.locator("select").select_option(label=f"{fiscal_year}年度")
        except:
            # 2. select要素ではなく、クリックしてリストが出るタイプの場合
            modal.locator(".ui-selectmenu-button").click() # プルダウン本体をクリック
            page.get_by_role("option", name=f"{fiscal_year}年度").click()
    
    print(f"Downloading data for FY{fiscal_year}")
    
    with page.expect_download(timeout=120_000) as dlinfo:
        # 「データダウンロード」ボタンをクリック
        modal.get_by_role("button", name="データダウンロード").click()
        
    download = dlinfo.value
    download.save_as(save_path)
    print(f"[OK] saved: {save_path}")

def process_csv(file_path):
    """CSVを読み込んで共通フォーマットに整形する"""
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(filepath_or_buffer=file_path, encoding="cp932")
    
    col_date = next(c for c in df.columns if "受渡日" in c)
    col_time = next(c for c in df.columns if "時刻コード" in c or "時間帯" in c)
    col_qty  = next(c for c in df.columns if "約定総量" in c)
    col_offer = next(c for c in df.columns if "売り入札量" in c)
    col_bid  = next(c for c in df.columns if "買い入札量" in c)
    col_tokyo= next(c for c in df.columns if "エリアプライス東京" in c or ("東京" in c and "プライス" in c))

    out = df.loc[:, [col_date, col_time, col_offer, col_bid, col_qty, col_tokyo]].copy()
    out.columns = ["date", "period", "offer_volume", "bid_volume", "traded_volume", "tokyo_price_jpy_per_kwh"]
    out["date"] = pd.to_datetime(out["date"])
    return out

def fetch_price():

    today = today_jst_date()
    seven_days_ago = today - timedelta(days=7)
    
    curr_fy = get_fiscal_year(today)
    prev_fy = get_fiscal_year(seven_days_ago)
    
    needs_prev_year = (curr_fy != prev_fy)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(accept_downloads=True, locale="ja-JP", timezone_id="Asia/Tokyo")
        page = ctx.new_page()
        page.goto("https://www.jepx.jp/electricpower/market-data/spot/", wait_until="domcontentloaded")

        # 日付の選択
        print("Push today cell")
        click_today_cell_only(page)

        # 「データダウンロード」をクリック → 年度選択ポップアップ → 下部のダウンロードをクリック
        print("Push data download")
        # まずページ本体の “開くボタン” をクリック（モーダルを開く）
        page.locator('#filter-section--type button.dl-button[data-dl="spot_summary"]').click(timeout=10_000)

        # モーダル表示を待つ
        modal = page.locator("#modal-box--spot_summary")
        modal.wait_for(state="visible", timeout=10_000)

        # 1. 今年度の分をダウンロード（プルダウン操作なし）
        download_spot_csv(page, modal, curr_fy, RAW_OUT_CURRENT, is_current=True)

        # 2. 年度跨ぎがある場合、前年度分もダウンロード（プルダウン操作あり）
        if needs_prev_year:
            # モーダルが一度閉じることが多いため、再表示を確認
            if not modal.is_visible():
                page.locator('#filter-section--type button.dl-button[data-dl="spot_summary"]').click()
                modal.wait_for(state="visible")
            
            download_spot_csv(page, modal, prev_fy, RAW_OUT_PREV, is_current=False)

        browser.close()

    # データ結合処理
    df_curr = process_csv(RAW_OUT_CURRENT)
    df_prev = process_csv(RAW_OUT_PREV) if needs_prev_year else None
    
    combined_df = pd.concat([df_prev, df_curr], ignore_index=True) if df_prev is not None else df_curr

    # 時系列処理
    combined_df["timestamp"] = (
        combined_df["date"] + (combined_df["period"] - 1).apply(lambda k: pd.Timedelta(minutes=30 * k))
    )
    combined_df.drop(columns=["period", "date"], inplace=True)
    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

    # フィルタリング
    filter_start = pd.Timestamp(seven_days_ago).tz_localize(None)
    filter_end = pd.Timestamp(today).tz_localize(None) + pd.Timedelta(hours=23, minutes=30)
    
    final_out = combined_df[combined_df["timestamp"].between(filter_start, filter_end)]
    final_out.reset_index(drop=True, inplace=True)
    print("final_out: \n", final_out)
    # 保存
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    final_out.to_parquet(PROC_OUT, index=False)
    print(f"[OK] Final data saved to {PROC_OUT}. Rows: {len(final_out)}")

if __name__ == "__main__":
    fetch_price()