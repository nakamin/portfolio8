from datetime import datetime, timedelta, timezone
import re, os, time
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

JST = timezone(timedelta(hours=9))
CACHE_DIR = "data/cache"
RAW_DIR = "data/raw"
DEBUG_DIR = "data/debug"

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

    # “当月かつ有効”な日セルの a/button を XPath で取得 → クリック
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
    except PWTimeout as e:
        # デバッグ出力して落とす
        page.screenshot(os.path.join(DEBUG_DIR, "calendar_page.png"), full_page=True)
        try: panel.first.screenshot(os.path.join(DEBUG_DIR, "calendar_panel.png"))
        except: pass
        page.context.tracing.stop(os.path.join(DEBUG_DIR, "trace.zip"))
        raise RuntimeError(
            "日セルをクリックできませんでした。debug: debug/calendar_page.png / calendar_panel.png / trace.zip"
        ) from e

def _set_by_label(scope, label_text: str, checked: bool):
    """
    - labelを手掛かりにチェック入力をチェック状態にする
    - scope: この範囲の中で要素を探す
    - label_text: 実際のラベルのテキスト
    - checked: 最終的にしたい状態（True: オン、False: オフ）
    """
    # .check-styleクラスが付いたlabelのうち、テキストが label_text を含む先頭を取る
    lab = scope.locator(f'label.check-style:has-text("{label_text}")').first
    lab.wait_for(state="visible", timeout=5000) # ラベル要素を取得して表示状態になるまで待機
    
    for_id = lab.get_attribute("for") # for属性から対応するinputを取得
    if not for_id:
        raise RuntimeError(f'label "{label_text}" に for 属性がありません')
    
    inp = scope.locator(f'input#{for_id}')
    # inp.wait_for(state="attached", timeout=5000) # 非表示でも存在すればOK

    # 現在値を見てズレていればクリック
    is_checked = inp.get_attribute("checked") is not None
    if is_checked != checked:
        lab.click(force=True, timeout=1000)

    # 検証（最大2回までリトライ）
    for _ in range(2):
        if inp.is_checked() == checked:
            return
        lab.click(force=True, timeout=1000)
        time.sleep(0.1)
    raise RuntimeError(f'"{label_text}" を {checked} にできませんでした')

def select_price_table_only(page):
    """
    約定価格（テーブル）
    - システムプライス: OFF
    - 東京: ON
    - 他エリア: OFF
    """

    # テーブル用のスコープを固定（グラフ側に触らない）
    scope = page.locator("#checkbox-area--table")
    scope.wait_for(state="visible", timeout=5000)

    # システムプライス OFF
    _set_by_label(scope, "システムプライス", False)

    # まず「全エリア」OFF（ONになっていることが多い）
    try:
        _set_by_label(scope, "全エリア", False)
    except Exception:
        pass

    # 各エリア
    for name in ["北海道","東北","東京","中部","北陸","関西","中国","四国","九州"]:
        _set_by_label(scope, name, (name == "東京"))

def select_quantity_table_only(page):
    """
    入札・約定量（テーブル側）:
    - 約定総量: ON
    - それ以外: OFF
    """

    # 入札・約定量の「テーブル側」スコープを取得
    scope = page.locator("#checkbox-volume--table")
    if scope.count() == 0:
        # 「入札・約定量」セクションの直下にある .checkbox-area--table を拾う
        section = page.locator("section.filter-sub-section").filter(
            has=page.locator('h4.filter-sub-section__ttl', has_text="入札・約定量")
        )
        scope = section.locator(".checkbox-area--table")
    scope.first.wait_for(state="visible", timeout=5000)
    scope = scope.first

    # まず全て OFF
    for label in ["約定総量","売り入札量","買い入札量","売りブロック入札量",
                "買いブロック入札量","売りブロック約定量","買いブロック約定量"]:
        try:
            _set_by_label(scope, label, False)
        except Exception:
            pass  # ないラベルは無視

    # 3つだけON
    for lb in ["約定総量","売り入札量","買い入札量"]:
        _set_by_label(scope, lb, True)

def fetch_price():
    raw_out  = os.path.join(RAW_DIR, "spot_tokyo_year.csv")
    proc_out = os.path.join(CACHE_DIR, "spot_tokyo_bf1w_tdy.parquet")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(accept_downloads=True, locale="ja-JP", timezone_id="Asia/Tokyo")
        page = ctx.new_page()
        page.goto("https://www.jepx.jp/electricpower/market-data/spot/", wait_until="domcontentloaded")

        # 日付の選択
        click_today_cell_only(page)

        # 表（テーブル）を選択
        print("Select table")
        try:
            page.get_by_role("button", name=re.compile("表|テーブル")).click(timeout=5000)
        except:
            pass  # 既に選択されている場合は無視

        # 「約定価格」：チェックを外して「東京」だけ残す
        print("Select area")
        select_price_table_only(page)
        # 「入札・約定量」：チェックを外して「約定総量」のみ
        print("Select a kind of price")
        select_quantity_table_only(page)

        # 「データダウンロード」をクリック → 年度選択ポップアップ → 下部のダウンロードをクリック
        print("Push data download")
        # まずページ本体の “開くボタン” をクリック（モーダルを開く）
        page.locator('#filter-section--type button.dl-button[data-dl="spot_summary"]').click(timeout=10_000)

        # モーダル表示を待つ
        modal = page.locator("#modal-box--spot_summary")
        modal.wait_for(state="visible", timeout=10_000)

        # モーダル内の “実行ボタン” をクリックして download を待つ（これでCSVが落ちる）
        with page.expect_download(timeout=120_000) as dlinfo:
            # name指定だと再び曖昧なので、モーダルを scope にして type=submit を叩く
            modal.locator('button.dl-button[type="submit"]').click()
        download = dlinfo.value
        download.save_as(raw_out)
        print(f"[OK] saved: {raw_out}")
        
        # 成功トレースも見たいときは保存
        page.context.tracing.stop(path=os.path.join(DEBUG_DIR, "trace_success.zip"))

    df = None
    df = pd.read_csv(filepath_or_buffer=raw_out, encoding="cp932")

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
    today = pd.Timestamp(datetime.today().date())
    before_1w = today - timedelta(days=7)
    tomorrow = today + timedelta(days=1)
    print(f"[FILTER] {before_1w} to {today}")
    
    final_out = out.sort_values(["date", "period"]).reset_index(drop=True)
    final_out.rename(columns={"date": "timestamp"}, inplace=True)
    final_out["timestamp"] = pd.to_datetime(final_out["timestamp"]) \
        + final_out["period"].apply(lambda x: pd.Timedelta(minutes=30 * x))
    final_out.drop(columns="period", inplace=True)
    print("final_outを一週間前から今日までにする前: \n", final_out)

    final_out = final_out[final_out["timestamp"].between(before_1w, tomorrow)]
    print("final_outを一週間前から今日までにする: \n", final_out)
    
    # 電力価格データの datetime 列を30分前にシフト
    final_out["timestamp"] = final_out["timestamp"] - pd.Timedelta(minutes=30)
    final_out = final_out.iloc[1:, :]
    final_out.reset_index(drop=True, inplace=True)
    print("シフト後: \n", final_out)
    
    final_out.to_parquet(proc_out, index=False)
    print(f"[OK] wrote filtered parquet (clean): {proc_out}  rows={len(final_out)}")

if __name__ == "__main__":
    fetch_price()