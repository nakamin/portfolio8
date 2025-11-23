import re, os, io, time
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests

# ===== 設定 =====
# EIA_API_KEY = os.getenv("EIA_API_KEY", "PUT_YOUR_API_KEY")
RAW_DIR = "data/raw"
PROC_DIR = "data/cache"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# キャッシュパス
FX_CACHE      = os.path.join(RAW_DIR, "fx_usdjpy.csv")
BRENT_CACHE   = os.path.join(RAW_DIR, "brent_daily.csv")
GAS_CACHE     = os.path.join(RAW_DIR, "gas_henryhub_daily.csv")
COAL_M_CACHE  = os.path.join(RAW_DIR, "coal_worldbank_monthly.csv")
ALL_CACHE  = os.path.join(RAW_DIR, "all_comodity.csv")

# ---- 共通ユーティリティ ----
def _req(url, params=None, timeout=30, max_retry=3):
    """
    - URLからjson形式でparamsで指定したデータを取得する
    """
    
    last = None
    for i in range(max_retry):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last = str(e)
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Request failed: {url} ({last})")

def _clip(df, date_col, start, end):
    """dfの日次範囲を指定する"""
    m = (df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))
    return df.loc[m].copy()

def _save_cache(df, path):
    """キャッシュとして指定したpathにcsvを保存する"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVE] df at {path}")

def _load_cache(path, date_col, start, end, source_label="cache"):
    """キャッシュとして保存したcsvを読み込む"""
    if not os.path.exists(path):
        raise RuntimeError("no cache")
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = _clip(df, date_col, start, end)
    if df.empty:
        raise RuntimeError("empty cache after clip")
    df["source"] = source_label
    
    print(f"[LOAD] df from {path}")
    return df

def _today_jst():
    JST = timezone(timedelta(hours=9))
    return datetime.now(JST).date()

def _range_to_today(days_back: int=10):
    """
    取得期間として、今日を基準に直近days_back日分を取る
    """
    t = _today_jst()
    start = t - pd.Timedelta(days=days_back)
    end   = t  # APIには「今日まで」で投げる（実際には最新営業日まで返ってくればOK）
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

# ===== 1) 為替（USD/JPY）：Frankfurter → yfinance → cache =====
def fetch_fx_usdjpy(start, end):
    """
    - 以下の優先度で為替価格を取得する
        - Frankfurter（APIキー不要）から為替価格を取得する
        - yfinanceから取得する
        - キャッシュを使用する
    """
    try:
        url = f"https://api.frankfurter.app/{start}..{end}"
        j = _req(url, params={"from": "USD", "to": "JPY"}).json()
        df = (pd.DataFrame(j["rates"]).T
                .rename_axis("date").reset_index())
        df["date"] = pd.to_datetime(df["date"])
        df["USDJPY"] = df["JPY"].astype(float)
        out = df[["date", "USDJPY"]].sort_values("date").reset_index(drop=True)
        if start:
            out = out[out["date"] >= pd.to_datetime(start)]
        out["source"] = "frankfurter"
        _save_cache(out, FX_CACHE)
        return out
    except Exception as e_fr:
        last_err = f"frankfurter failed: {e_fr}"

    # yfinance の通貨ペア（USD/JPY は "JPY=X"）
    try:
        import yfinance as yf
        tkr = yf.Ticker("JPY=X")
        hist = tkr.history(start=start, end=end, interval="1d", actions=False, auto_adjust=False)
        if hist is None or hist.empty:
            out = (hist.reset_index()
                    .rename(columns={"Date":"date","Close":"USDJPY"})
                    [["date","USDJPY"]].dropna().sort_values("date"))
            out["source"] = "yfinance"
            _save_cache(out, FX_CACHE)
            return out
        last_err = "yfinance empty"
    except Exception as e_yf:
        last_err = f"yfinance failed: {e_yf}"

    # cache
    return _load_cache(FX_CACHE, "date", start, end, "cache")

# ===== 2) Brent：EIA → yfinance（BZ=F）→ cache =====
def fetch_brent_daily(start, end, api_key=None):
    """
    - 以下の優先度で原油価格を取得する
        - EIA Europe Brent Spot Price FOB (Daily)（APIキー必要）から為替価格を取得する
        - yfinanceから取得する
        - キャッシュを使用する
    """

    try:
        if api_key is None:
            api_key = os.getenv("EIA_API_KEY", "PUT_YOUR_API_KEY")

        if not api_key or api_key == "PUT_YOUR_API_KEY":
            raise RuntimeError("EIA_API_KEY not set")

        # v2 の seriesid 互換ルート
        url = "https://api.eia.gov/v2/seriesid/PET.RBRTE.D"
        params = {"api_key": api_key}
        if start:
            params["start"] = pd.to_datetime(start).strftime("%Y-%m-%d")
        if end:
            params["end"]   = pd.to_datetime(end).strftime("%Y-%m-%d")

        j = _req(url, params=params).json()
        
        if "error" in j:
            raise RuntimeError(j["error"])
        data = j.get("response", {}).get("data", [])
        if not data:
            raise RuntimeError("EIA empty data")
        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "period" in df.columns:
            df["date"] = pd.to_datetime(df["period"])
        else:
            raise RuntimeError("date/period missing")
        val_col = "value" if "value" in df.columns else df.columns.difference(["date","period"]).tolist()[0]
        out = (df.rename(columns={val_col: "brent"})
                [["date","brent"]].sort_values("date").reset_index(drop=True))
        if start:
            out = out[out["date"] >= pd.to_datetime(start)]
            out.reset_index(drop=True, inplace=True)

        out["source"] = "EIA"
        _save_cache(out, BRENT_CACHE)
        return out
    except Exception as e_eia:
        last_err = f"EIA failed: {e_eia}"
        print(last_err)

    # yfinance：ICE Brent front-month futures
    try:
        import yfinance as yf
        tkr = yf.Ticker("BZ=F")  # ICE Brent front-month
        hist = tkr.history(start=start, end=end, interval="1d", actions=False, auto_adjust=False)
        if hist is not None and not hist.empty:
            out = (hist.reset_index()
                    .rename(columns={"Date": "date", "Close": "brent"})
                    [["date", "brent"]]
                    .dropna()
                    .sort_values("date")
                    .reset_index(drop=True))
            out["source"] = "yfinance"
            _save_cache(out, BRENT_CACHE)
            return out
        last_err = "yfinance empty"
    except Exception as e_yf:
        last_err = f"yfinance failed: {e_yf}"

    # cache
    return _load_cache(BRENT_CACHE, "date", start, end, "cache")

# ===== 3) Henry Hub：EIA → yfinance（NG=F）→ cache =====
def fetch_gas_henryhub_daily(start, end, api_key=None):
    """
    - 以下の優先度で天然ガス価格を取得する
        - EIA Henry Hub spot daily（APIキー必要）から為替価格を取得する
        - yfinanceから取得する
        - キャッシュを使用する
    """
    try:
        if api_key is None:
            api_key = os.getenv("EIA_API_KEY", "PUT_YOUR_API_KEY")

        if not api_key or api_key == "PUT_YOUR_API_KEY":
            raise RuntimeError("EIA_API_KEY not set")

        url = "https://api.eia.gov/v2/seriesid/NG.RNGWHHD.D"

        params = {"api_key": api_key}
        if start:
            params["start"] = pd.to_datetime(start).strftime("%Y-%m-%d")
        if end:
            params["end"]   = pd.to_datetime(end).strftime("%Y-%m-%d")
        j = _req(url, params=params).json()
        
        if "error" in j:
            raise RuntimeError(j["error"])

        data = j.get("response", {}).get("data", [])
        if not data:
            raise RuntimeError("EIA empty data")
        
        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "period" in df.columns:
            df["date"] = pd.to_datetime(df["period"])
        else:
            raise RuntimeError("date/period missing")
        
        val_col = "value" if "value" in df.columns else df.columns.difference(["date","period"]).tolist()[0]
        out = (df.rename(columns={val_col: "henry_hub"})
                [["date","henry_hub"]].sort_values("date").reset_index(drop=True))
        if start:
            out = out[out["date"] >= pd.to_datetime(start)]
            out.reset_index(drop=True, inplace=True)
            
        out["source"] = "EIA"
        _save_cache(out, GAS_CACHE)
        return out
    except Exception as e_eia:
        last_err = f"EIA failed: {e_eia}"
        print(last_err)

    # yfinance：NYMEX Natural Gas futures
    try:
        import yfinance as yf

        tkr = yf.Ticker("NG=F")  # NYMEX NatGas front-month
        hist = tkr.history(start=start, end=end, interval="1d", actions=False, auto_adjust=False)
        if hist is not None and not hist.empty:
            out = (hist.reset_index()
                    .rename(columns={"Date": "date", "Close": "henry_hub"})
                    [["date", "henry_hub"]]
                    .dropna()
                    .sort_values("date")
                    .reset_index(drop=True))
            out["source"] = "yfinance"
            _save_cache(out, GAS_CACHE)
            return out
        last_err = "yfinance empty"
    except Exception as e_yf:
        last_err = f"yfinance failed: {e_yf}"

    # cache
    return _load_cache(GAS_CACHE, "date", start, end, "cache")

# ===== 4) 石炭：World Bank（月次）→ cache =====
def fetch_coal_worldbank_monthly(start, end):
    """
    - 以下の優先度で石炭価格を取得する
        - Pink Sheet: Monthly Prices（APIキー不要）から為替価格を取得する
        - yfinanceから取得する
        - キャッシュを使用する
    """
    try: 
        url = ("https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/related/CMO-Historical-Data-Monthly.xlsx")
        r = _req(url)
        wb = pd.read_excel(io.BytesIO(r.content), sheet_name="Monthly Prices", header=None)
        code_row, coal_col = None, None
        for r in range(0, 20):  # 先頭20行程だけ読み込む
            row_vals = wb.iloc[r].astype(str).tolist()
            if "Coal, Australian" in row_vals:
                code_row = r
                coal_col = row_vals.index("Coal, Australian")
                break
        if coal_col is None:
            raise RuntimeError("Coal, Australian column is not found.")

        start_row = code_row + 2 # データの始まり行

        # 4) 月列（先頭列）＋ 豪州炭列を抽出
        coal = wb.iloc[start_row:, [0, coal_col]].copy()
        coal.columns = ["month_raw", "coal_aus"]
        
        # 5) '1960M01' → Timestamp(1960-01-01) へ
        def parse_mon(s):
            if pd.isna(s):
                return pd.NaT
            s = str(s).strip()
            m = re.match(r"^\s*(\d{4})[Mm](\d{1,2})(?:\D.*)?$", s)  # 後ろに注記が付いてもOK
            if m:
                y = int(m.group(1)); mm = int(m.group(2))
                if 1 <= mm <= 12:
                    return pd.Timestamp(y, mm, 1)
            return pd.to_datetime(s, errors="coerce")

        coal["month"] = coal["month_raw"].map(parse_mon)
        coal = coal.dropna(subset=["month"]).drop(columns=["month_raw"])

        # 6) 数値化＆整形
        coal["coal_aus"] = pd.to_numeric(coal["coal_aus"], errors="coerce")
        coal = coal.dropna(subset=["coal_aus"]).sort_values("month").reset_index(drop=True)

        today = _today_jst()
        today_ts = pd.to_datetime(today)
        today_period = today_ts.to_period("M")
        prev_month_start = (today_period - 1).to_timestamp()
        print("last_month: ", prev_month_start)
        coal_filtered = coal[coal["month"] >= prev_month_start]
        
        if coal_filtered.empty:
            # monthでソートして末尾1行
            coal = coal.sort_values("month")
            coal_filtered = coal.tail(1)

        coal_filtered["source"] = "worldbank"
        coal_filtered = coal_filtered[["month", "coal_aus", "source"]]
        _save_cache(coal_filtered, COAL_M_CACHE)
        return coal_filtered

    except Exception as e_wb:
        print(f"World Bank failed: {e_wb}")


def coal_monthly_to_daily_ffill(coal_m_df, start, end):
    """
    - 月次価格を日次価格にする
    """
    daily_idx = pd.date_range(start, end, freq="D")
    coal_d = (coal_m_df.set_index("month")
            .reindex(daily_idx, method="ffill")
            .rename_axis("date").reset_index())
    coal_d = coal_d[["date","coal_aus"]]
    coal_d["source"] = coal_m_df["source"].iloc[-1] if not coal_m_df.empty else "cache"
    return coal_d

def expand_daily_to_30min(df_daily, value_cols):
    expanded = []
    for _, row in df_daily.iterrows():
        for i in range(48):
            dt = pd.to_datetime(row["timestamp"]) + pd.Timedelta(minutes=30 * i)
            entry = {"timestamp": dt}
            for col in value_cols:
                entry[col] = row[col]
            expanded.append(entry)
    return pd.DataFrame(expanded)

# ===== 結合（学習/リアルタイム共通） =====
def build_feature_table(start, end):
    """
    - すべてdfを統合する
    """
    fx    = fetch_fx_usdjpy(start, end)                     # USDJPY
    print("fx: \n", fx.head())
    brent = fetch_brent_daily(start, end, None)      # brent（スポットor先物）
    print("brent: \n", brent.head())
    gas   = fetch_gas_henryhub_daily(start, end, None)
    print("gas: \n", gas.head())
    coalm = fetch_coal_worldbank_monthly(start, end)
    print("coalm: \n", coalm.head())
    coal  = coal_monthly_to_daily_ffill(coalm, start, end)
    print("coal: \n", coal.head())

    for d in (fx, brent, gas, coal):
        d["date"] = pd.to_datetime(d["date"]).dt.date

    base = pd.DataFrame({
    "date": pd.date_range(start, end, freq="D")})
    base["date"] = pd.to_datetime(base["date"]).dt.date
    
    df = (base.merge(fx[["date","USDJPY"]], on="date", how="left")
               .merge(brent[["date","brent"]], on="date", how="left")
               .merge(gas[["date","henry_hub"]], on="date", how="left")
               .merge(coal[["date","coal_aus"]], on="date", how="left"))

    # 補完（前方→後方）
    df = df.sort_values("date")
    df.rename(columns={"date": "timestamp"}, inplace=True)
    for col in ["USDJPY","brent","henry_hub","coal_aus"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
            
    print("df(commodities combined): \n", df)
    
    _save_cache(df, ALL_CACHE)
    latest_df = df.tail(8).reset_index(drop=True)
    latest_df["timestamp"] = pd.to_datetime(latest_df["timestamp"], errors="coerce")
    print("latent_df: \n", latest_df)
    
    out_path = os.path.join(PROC_DIR, "fx_commodity_day.parquet")
    latest_df.to_parquet(out_path)
    print(f"[OK] market unified: {out_path}")
    

    extended_market = expand_daily_to_30min(latest_df, ["USDJPY", "brent", "henry_hub", "coal_aus"])
    last_row = extended_market.iloc[-1].copy()
    # 開始時刻（最後の行の日時）
    start_time = last_row["timestamp"]
    # 新しい行を格納するリスト
    new_rows = []
    # 30分刻みで6日分（48 × 6 = 336回）
    for i in range(1, 48 * 6 + 1):
        new_row = last_row.copy()
        new_row["timestamp"] = start_time + pd.Timedelta(minutes=30 * i)
        new_rows.append(new_row)

    # DataFrame化して結合
    df_future = pd.DataFrame(new_rows)
    extended_market = pd.concat([extended_market, df_future], ignore_index=True)
    print("market_extended: \n", extended_market)
    
    out_path = os.path.join(PROC_DIR, "fx_commodity_30min_af1w.parquet")
    extended_market.to_parquet(out_path)
    print(f"[OK] extended_market unified: {out_path}")

def fetch_market():
    
        # 今日までのデータを取得する
        start, end = _range_to_today(days_back=10)
        build_feature_table(start, end)

if __name__ == "__main__":
    fetch_market()