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

RAW_OUT  = RAW_DIR / "spot_tokyo_year.csv"
PROC_OUT = CACHE_DIR / "spot_tokyo_bf1w_tdy.parquet"

URL = "https://www.jepx.jp/electricpower/market-data/spot/"

def today_jst_date():
    return datetime.now(JST).date()

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

def fetch_price():

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

        # モーダル内の “実行ボタン” をクリックして download を待つ（これでCSVが落ちる）
        with page.expect_download(timeout=120_000) as dlinfo:
            modal.locator('button.dl-button[type="submit"]').click()
        download = dlinfo.value
        download.save_as(RAW_OUT)
        print(f"[OK] saved: {RAW_OUT}")
        
        # 成功トレースも見たいときは保存
        page.context.tracing.stop(path=os.path.join(DEBUG_DIR, "trace_success.zip"))

    df = None
    df = pd.read_csv(filepath_or_buffer=RAW_OUT, encoding="cp932")

    # 列名
    col_date = next(c for c in df.columns if "受渡日" in c)
    col_time = next(c for c in df.columns if "時刻コード" in c or "時間帯" in c)
    col_qty  = next(c for c in df.columns if "約定総量" in c)
    col_offer  = next(c for c in df.columns if "売り入札量" in c)
    col_bid  = next(c for c in df.columns if "買い入札量" in c)
    col_tokyo= next(c for c in df.columns if "エリアプライス東京" in c or ("東京" in c and "プライス" in c))

    out = df.loc[:, [col_date, col_time, col_offer, col_bid, col_qty, col_tokyo]].copy()
    out.columns = ["date", "period", "offer_volume", "bid_volume", "traded_volume", "tokyo_price_jpy_per_kwh"]
    out["date"] = pd.to_datetime(out["date"])

    # 過去1週間分だけ切り出して保存
    today = pd.Timestamp(datetime.now(JST).date())
    before_1w = today - timedelta(days=7)
    end_inclusive = today + pd.Timedelta(hours=23, minutes=30)
    print(f"[FILTER] {before_1w} to {end_inclusive}")
    
    final_out = out.sort_values(["date", "period"]).reset_index(drop=True)
    final_out.rename(columns={"date": "timestamp"}, inplace=True)
    final_out["timestamp"] = (
        final_out["timestamp"]
        + (final_out["period"] - 1).apply(lambda k: pd.Timedelta(minutes=30 * k))
    )
    final_out.drop(columns="period", inplace=True)
    print("ダウンロード結果: \n", final_out)

    final_out = final_out[final_out["timestamp"].between(before_1w, end_inclusive, inclusive="both")]
    final_out.reset_index(drop=True, inplace=True)
    print("final_outを一週間前から今日までにする: \n", final_out)
    
    final_out.to_parquet(PROC_OUT, index=False)
    print(f"[OK] wrote filtered parquet (clean): {PROC_OUT}  rows={len(final_out)}")

if __name__ == "__main__":
    fetch_price()