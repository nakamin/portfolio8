import re, requests
from datetime import datetime, timedelta, timezone
import pandas as pd
from io import StringIO
from pathlib import Path
import os


JST = timezone(timedelta(hours=9))

PAGE_URL = "https://www.tepco.co.jp/forecast/html/area_jukyu-j.html"
CACHE_DIR = Path("data/cache")
DATA_DIR = Path("data")

ACTUAL_PATH = DATA_DIR / "actual.parquet"
DEMAND_PATH = CACHE_DIR / "demand_bf1w_ytd.parquet"

UPDATE_ACTION_DAY = 8

columns = [
    "date", "time", "demand", "nuclear",
    "lng", "coal", "oil", "th_other",
    "hydro", "geothermal", "biomass",
    "pv", "pv_curtailed", "wind", "wind_curtailed",
    "pstorage", "battery", "tie", "misc", "total"
]

def _get_month_csv_url(session: requests.Session, target_ym) -> str:
    """ページから当月CSVのURLを拾う（a要素のhrefに .csv が含まれるもの）"""
    
    html = session.get(PAGE_URL, timeout=30).text
    # CSVリンクをすべて抽出
    csv_links = re.findall(r'href="([^"]*eria_jukyu_\d{6}_\d{2}\.csv)"', html, re.I)

    # 昨日の月に一致するリンクだけを抽出
    filtered = [link for link in csv_links if target_ym in link]
    print(filtered)

    # 完全なURLに変換
    def complete_url(link):
        if link.startswith("//"):
            return "https:" + link
        elif link.startswith("/"):
            return "https://www.tepco.co.jp" + link
        else:
            return link

    return [complete_url(link) for link in filtered]

def read_demand_csv(content: str) -> pd.DataFrame:
    # 1行目を読み込んで、混在しているか確認
    first_line =  content.splitlines()[0]

    # 1行目を分割して、カラム数より多ければ混在していると判断
    first_split = first_line.strip().split(",")
    if len(first_split) > len(columns):
        print("Misalignment is found")
        # 最初のデータ行を抽出（カラム数分以降）
        first_data = first_split[len(columns):]
        # 2行目以降をDataFrameとして読み込む
        df = pd.read_csv(StringIO(content), encoding="MacRoman", skiprows=1, header=None)
        df.columns = columns
        # 最初のデータ行を先頭に挿入
        first_row = pd.DataFrame([first_data], columns=columns)
        df = pd.concat([first_row, df], ignore_index=True)
    else:
        # 通常通り読み込む
        df = pd.read_csv(StringIO(content), encoding="MacRoman", skiprows=2, header=None) # requestするときはずれがないのでこっちになる
        df.columns = columns
    
    print("df: ", df)
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["realized_demand"] = df["demand"].astype("float64")
    
    df.drop(columns=["date", "time", "demand"], inplace=True)

    return df

def update_actual_last_month(last_month_df):
    
    if last_month_df.empty:
        print("[update_actual] last_month_df is empty → skip")
        return
    
    print("last_month_df: ", last_month_df)
    
    # timestamp を index にして、長期用フォーマットに揃える
    last_month_df = last_month_df.set_index("timestamp").sort_index()

    # 既存 actual を読み込んで結合
    if ACTUAL_PATH.exists():
        actual = pd.read_parquet(ACTUAL_PATH)

        combined = pd.concat([actual, last_month_df])
        # 同じ timestamp があれば新しい方を残す
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = last_month_df
    
    combined.drop(columns=["date", "time"], inplace=True, errors="ignore")
    print("combined: \n", combined)

    # 書き出し
    combined.to_parquet(ACTUAL_PATH)
    print(f"[update_actual] updated {ACTUAL_PATH} (rows={len(combined)})")

def fetch_demand():
    """
    - 1-7日: 前月と当月のCSVを結合して取得
    - 8日以降: 当月のCSVのみ取得
    - 8日当日: 前月分を確定版としてキャッシュ保存 (update_actual_last_month)
    """

    # 昨日の日付起算で当月を取得
    today = datetime.now(JST)
    print(f"Today: {today}")
    
    this_month_ym = today.strftime("%Y%m")
    last_month_ym = (today.replace(day=1) - timedelta(days=1)).strftime("%Y%m")

    is_early_month = today.day <= 7
    is_update_day = today.day == 8

    # 取得候補のリスト（1-7日なら [前月, 当月]、8日以降なら [当月]）
    target_yms = [last_month_ym, this_month_ym] if is_early_month else [this_month_ym]
    
    dfs = []
    with requests.Session() as s:
        for ym in target_yms:
            print(f"Searching for YM: {ym}")
            csv_urls = _get_month_csv_url(session=s, target_ym=ym)
            
            if csv_urls:
                print(f"Fetching: {csv_urls[0]}")
                response = s.get(csv_urls[0], timeout=30)
                response.encoding = "MacRoman"
                content = response.content.decode("MacRoman")
                
                month_df = read_demand_csv(content)
                if month_df is not None:
                    dfs.append(month_df)
                    
                    # 8日の場合、前月分を取得したタイミングでキャッシュ更新
                    if is_update_day and ym == last_month_ym:
                        print(f"Today is 8th. Updating actual cache for {last_month_ym}")
                        update_actual_last_month(month_df)

        # 8日に当月分しかリストにない場合、前月分を別途取得してキャッシュ更新
        if is_update_day and len(dfs) > 0 and not any(d['timestamp'].dt.strftime('%Y%m').iloc[0] == last_month_ym for d in dfs if not d.empty):
            # 8日はtarget_ymsが[this_month]のみになるため、ここで前月分を処理
            prev_urls = _get_month_csv_url(session=s, target_ym=last_month_ym)
            if prev_urls:
                resp = s.get(prev_urls[0], timeout=30)
                prev_content = resp.content.decode("MacRoman")
                prev_df = read_demand_csv(prev_content)
                update_actual_last_month(prev_df)

    # データの結合
    if not dfs:
        print("No CSV data found.")
        return None

    df = pd.concat(dfs, ignore_index=True)
    print("[Concat]dfs: \n", df)
    
    # 1週間前から昨日までを抽出
    before_1w = (today - timedelta(days=7)).date()
    yesterday = (today - timedelta(days=1)).date()
    print(f"[Filter] from {before_1w} to {yesterday}")
    prev = df[(df["timestamp"].dt.date >= before_1w)&(df["timestamp"].dt.date <= yesterday)].copy().reset_index(drop=True)

    if prev.empty:
        print("[fetch_demand] prev is empty – fallback: use last available week pattern")
        
        if df.empty:
            print("[fetch_demand] df itself is empty – cannot build fallback")
            return
    
        last_ts = df["timestamp"].max()
        last_date = last_ts.date()
        hist_start = last_date - timedelta(days=6)
    
        hist = df[
        (df["timestamp"].dt.date >= hist_start) &
        (df["timestamp"].dt.date <= last_date)
        ].copy()
        
        if hist.empty:
            print("[fetch_demand] hist (last week) is empty – cannot build fallback")
            return
        src_start = hist["timestamp"].dt.date.min()
        
        offset_days = (before_1w - src_start).days
        print(f"[fetch_demand] shifting last week by {offset_days} days | start: {src_start} end: {last_date}")
        
        hist["timestamp"] = hist["timestamp"] + pd.Timedelta(days=offset_days) # 各datetimeにoffset_daysぶんを足す処理

        # 目的の期間だけに絞り直す
        prev = hist[
            (hist["timestamp"].dt.date >= before_1w) &
            (hist["timestamp"].dt.date <= yesterday)
        ].copy()
        prev = hist.reset_index(drop=True)

    print("final_out: \n", prev)

    prev.to_parquet(DEMAND_PATH, index=False)
    print(f"[SAVE] before_1w demand: {DEMAND_PATH}")
    
if __name__ == "__main__":
    fetch_demand()