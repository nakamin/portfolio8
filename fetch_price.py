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
    except PlaywrightTimeoutError as e:
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
    - タイムアウト時は警告ログを出してスキップする
    """

    # 1) ラベル取得
    lab = scope.locator(f'label.check-style:has-text("{label_text}")').first
    try:
        lab.wait_for(state="visible", timeout=15_000)
    except PlaywrightTimeoutError:
        print(f'[WARN] label "{label_text}" が見つからない or visible にならない → スキップ')
        return

    # 2) for 属性 → input の id を取得
    for_id = lab.get_attribute("for")
    if not for_id:
        print(f'[WARN] label "{label_text}" に for 属性がない → スキップ')
        return

    inp = scope.locator(f'input#{for_id}')
    try:
        # 存在すればOK
        inp.wait_for(state="attached", timeout=15_000)
    except PlaywrightTimeoutError:
        print(f'[WARN] input#{for_id} が DOM に attach されない → スキップ')
        return

    # 3) 現在の状態（checked 属性の有無）を確認
    def _attr_checked() -> bool:
        attr = inp.get_attribute("checked")
        return attr is not None

    try:
        current = _attr_checked()
    except PlaywrightTimeoutError:
        print(f'[WARN] input#{for_id} の checked 属性取得で timeout → スキップ')
        return

    # すでに望む状態なら何もしない
    if current == checked:
        return

    # 4) クリックして状態を変更（最大数回リトライ）
    for attempt in range(3):
        try:
            lab.click(force=True, timeout=2_000)
        except PlaywrightTimeoutError:
            print(f'[WARN] label "{label_text}" click timeout (attempt={attempt+1})')
            continue

        time.sleep(0.2)  # DOM 更新待ち

        try:
            if _attr_checked() == checked:
                return
        except PlaywrightTimeoutError:
            print(f'[WARN] input#{for_id} attr check timeout after click (attempt={attempt+1})')
            continue

    print(f'[WARN] "{label_text}" を {checked} にできなかったが処理は継続する')


def select_price_table_only(page):
    """
    約定価格（テーブル）
    - システムプライス: OFF
    - 東京: ON
    - 他エリア: OFF
    """

    # テーブル用のスコープを固定（グラフ側に触らない）
    scope = page.locator("#checkbox-area--table")
    
    try:
        scope.wait_for(state="attached", timeout=15000)
    except PlaywrightTimeoutError:
        print("[WARN] #checkbox-area--table が visible にならないため、"
            "エリア/システムのチェック切り替えをスキップします。")
        return
    
    # システムプライス OFF
    try:
        _set_by_label(scope, "システムプライス", False)
    except Exception:
        pass
    # まず「全エリア」OFF（ONになっていることが多い）
    try:
        _set_by_label(scope, "全エリア", False)
    except Exception:
        pass

    # 各エリア
    for name in ["北海道","東北","東京","中部","北陸","関西","中国","四国","九州"]:
        _set_by_label(scope, name, (name == "東京"))

def select_quantity_table_only(page) -> bool:
    """
    入札・約定量（テーブル側）:
    - 約定総量: ON
    - それ以外: OFF
    """
    page.wait_for_load_state("networkidle")  # ページ全体が落ち着くまで待機

    # 入札・約定量の「テーブル側」スコープを取得
    scope = page.locator("#checkbox-volume--table")
    if scope.count() == 0:
        # 「入札・約定量」セクションの直下にある .checkbox-area--table を拾う
        section = page.locator("section.filter-sub-section").filter(
            has=page.locator('h4.filter-sub-section__ttl', has_text="入札・約定量")
        )
        if section.count() == 0:
            print("[WARN] 入札・約定量セクションが見つからない")
            return False
        scope = section.locator(".checkbox-area--table").first
    else:
        scope = scope.first

    try:
        scope.wait_for(state="visible", timeout=15000)
    except PlaywrightTimeoutError:
        print("[WARN] 入札・約定量のcheckbox-area--table が visible にならなかった")
        return False

    # まず全て OFF
    for label in [
        "約定総量","売り入札量","買い入札量","売りブロック入札量",
        "買いブロック入札量","売りブロック約定量","買いブロック約定量"
    ]:
        try:
            _set_by_label(scope, label, False)
        except Exception:
            # 存在しないラベルは無視
            pass

    # 3つだけ ON
    for lb in ["約定総量","売り入札量","買い入札量"]:
        _set_by_label(scope, lb, True)

    return True

def fallback_price_from_yesterday():
    """昨日の price cache を 1 日ずらして今日のデータとして再保存"""

    if not PROC_OUT.exists():
        print(f"[ERROR] fallback: {PROC_OUT} が存在しません")
        return False

    df = pd.read_parquet(PROC_OUT)
    if "timestamp" not in df.columns:
        print("[ERROR] fallback: timestamp がありません")
        return False

    df["timestamp"] = pd.to_datetime(df["timestamp"]) + timedelta(days=1)
    df.to_parquet(PROC_OUT, index=False)

    print("[INFO] fallback: yesterday's price used instead (shifted +1 day)")
    return True

def fetch_price():

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
        ok = select_quantity_table_only(page)
        
        if not ok:
            print("[WARN] price UI が取れないので fallback に切り替え")
            if fallback_price_from_yesterday():
                return
            else:
                print("[ERROR] fallback にも失敗 → 価格更新スキップ")
                return

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
    
    final_out.to_parquet(PROC_OUT, index=False)
    print(f"[OK] wrote filtered parquet (clean): {PROC_OUT}  rows={len(final_out)}")

if __name__ == "__main__":
    fetch_price()