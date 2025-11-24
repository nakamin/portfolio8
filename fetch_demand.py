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

def read_csv(path: str) -> pd.DataFrame:
    # 1行目を読み込んで、混在しているか確認
    with open(path, encoding="MacRoman") as f:
        first_line = f.readline()
        rest = f.readlines()

    # 1行目を分割して、カラム数より多ければ混在していると判断
    first_split = first_line.strip().split(",")
    if len(first_split) > len(columns):
        # 最初のデータ行を抽出（カラム数分以降）
        first_data = first_split[len(columns):]
        # 2行目以降をDataFrameとして読み込む
        df = pd.read_csv(path, encoding="MacRoman", skiprows=1, header=None)
        df.columns = columns
        # 最初のデータ行を先頭に挿入
        first_row = pd.DataFrame([first_data], columns=columns)
        df = pd.concat([first_row, df], ignore_index=True)
    else:
        # 通常通り読み込む
        df = pd.read_csv(path, encoding="MacRoman", skiprows=1, header=None)
        df.columns = columns

    return df

def update_actual_last_month(df, today, actuion_day):
    # 毎月8日以外は何もしない
    if today.day != actuion_day:
        print(f"[update_actual] today.day != {actuion_day} → skip")
        return
    
    # 先月の 1日〜末日を計算
    first_this_month = today.date().replace(day=1)
    last_month_end = first_this_month - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    
    print(f"[update_actual] last month = {last_month_start} 〜 {last_month_end}")

    # 先月分だけ切り出し
    mask = (
        (df["timestamp"].dt.date >= last_month_start)
        & (df["timestamp"].dt.date <= last_month_end)
    )
    last_month_df = df.loc[mask].copy()
    
    if last_month_df.empty:
        print("[update_actual] last_month_df is empty → skip")
        return
    
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
    
    print("combined: \n", combined)

    # 書き出し
    combined.to_parquet(ACTUAL_PATH)
    print(f"[update_actual] updated {ACTUAL_PATH} (rows={len(combined)})")

def fetch_demand():
    """当月CSVをDLして前日分とバンド（同月同時刻のmin/max）を用意する"""

    # 昨日の日付起算で当月を取得
    today = datetime.now(JST)
    print(f"Today: {today}")
    before_1w = (today - timedelta(days=7)).date()
    yesterday = (today - timedelta(days=1)).date()
    target_ym = before_1w.strftime("%Y%m")
    print(f"Target date: {before_1w} (YM={target_ym})")

    with requests.Session() as s:
        # 当月を探す
        csv_url = _get_month_csv_url(session=s, target_ym=target_ym)
        # csv_url = None
        # 前月を探す
        if not csv_url:
            prev_month_ym = (before_1w - timedelta(days=31)).strftime("%Y%m")
            print(f"{target_ym} is not found.Let's search {prev_month_ym}")
            csv_url = _get_month_csv_url(session=s, target_ym=prev_month_ym)
        if not csv_url:
            print("We could not find prev_month_ym csv")
        
        print("csv_url: ", csv_url)
        response = s.get(csv_url[0], timeout=30)
        response.encoding = "MacRoman"  # TEPCOのCSVはShift-JIS
        content = response.content.decode("MacRoman")  # bytes → str

        # CSV文字列をファイルのように扱う
        df = pd.read_csv(StringIO(content), skiprows=2, header=None)
        df.columns = columns
    
    if df is None:
        print("We could not get csv itself, so update the demand.")
        return 
    
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["realized_demand"] = df["demand"].astype("float64")
    
    df.drop(columns=["date", "time", "demand"], inplace=True)
    update_actual_last_month(df, today, 8)
    
    # 1週間前から昨日までを抽出
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