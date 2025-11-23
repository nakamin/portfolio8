import os, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

CACHE_DIR = "data/cache"

# Open-Meteoから取得する変数
OM_HOURLY_VARS = ",".join([
    "temperature_2m",
    "wind_speed_10m",
    "sunshine_duration",   # 秒/時
    "shortwave_radiation", # W/m^2（時間平均）
    "relative_humidity_2m" # 湿度
])

def _today_jst():
    JST = timezone(timedelta(hours=9))
    return datetime.now(JST).date()

def _hourly_to_30min(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    - 一時間単位→30分単位に変換
    - 温度/風/放射は線形、日照は半分配分で変換
    """
    df = df_hourly.set_index("timestamp").sort_index()

    # ダミー行を追加して resample 範囲を広げる
    last_time = df.index.max()
    dummy_time = last_time + pd.Timedelta(hours=1)
    dummy_row = pd.DataFrame({col: np.nan for col in df.columns}, index=[dummy_time])
    df = pd.concat([df, dummy_row])

    # 線形内挿
    cols_lin = [c for c in ["temperature_2m","wind_speed_10m","shortwave_radiation", "relative_humidity_2m"] if c in df.columns]
    out = df[cols_lin].resample("30min").interpolate("time")

    # 日照（秒/時）→ 30分に変換（半分にする）
    if "sunshine_duration" in df.columns:
        out["sunshine_duration"] = df["sunshine_duration"].resample("30min").interpolate("time") * 0.5

    out.reset_index(drop=False, inplace=True)
    out.rename(columns={"index": "timestamp"}, inplace=True)
    return out

def _fetch_openmeteo_hourly(lat: float, lon: float, tz: str,
                            past_days: int, forecast_days: int) -> pd.DataFrame:
    """
    Open-MeteoのForecast APIを1回叩いて、直近の実績＋先の予報をまとめて取得（一時間単位）
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,           # 返却時刻をJSTにする
        "hourly": OM_HOURLY_VARS,
        "past_days": past_days,   # 直近の実績
        "forecast_days": forecast_days,  # 先の予報（日数）
    }
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    j = r.json()
    h = j.get("hourly", {})
    if not h or "time" not in h:
        raise RuntimeError("Open-Meteo hourly is empty")
    df = pd.DataFrame(h)
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for c in ["temperature_2m","wind_speed_10m","sunshine_duration","shortwave_radiation", "relative_humidity_2m"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)

def build_openmeteo_daily(lat: float, lon: float, tz: str = "Asia/Tokyo",
                          past_days: int = 1, forecast_days: int = 7):
    """
    - hourlyを一発取得（過去+予報）
    - 今日00:00を境に「実績相当」と「予報」に分割
    - 各々30分化して保存
    """
    today = _today_jst()
    cut = pd.Timestamp(f"{today} 00:00:00")  # JST基準（tzで返ってくる）

    hourly = _fetch_openmeteo_hourly(lat, lon, tz, past_days, forecast_days)
    print("hourly: \n", hourly)
    # 30分化
    hourly_30 = _hourly_to_30min(hourly)
    print("hourly_30: \n", hourly_30)
    
    # 実績（過去）と予報に分割
    actual_30 = hourly_30[hourly_30["timestamp"] < cut].copy()
    fcst_30   = hourly_30[hourly_30["timestamp"] >= cut].copy()

    # ラベル付け
    actual_30["source"] = "Open-Meteo"
    actual_30["is_forecast"] = 0
    fcst_30["source"] = "Open-Meteo"
    fcst_30["is_forecast"] = 1

    # 欠損がある場合は前方補完
    for df in (actual_30, fcst_30):
        for c in ["temperature_2m","wind_speed_10m","shortwave_radiation","sunshine_duration", "relative_humidity_2m"]:
            if c in df.columns:
                df[c] = df[c].fillna(method="ffill", limit=2)

    out_path = os.path.join(CACHE_DIR, "weather_bf1w_af1w.parquet")

    df = pd.concat([actual_30, fcst_30], ignore_index=True).sort_values("timestamp")
    df = df.sort_values(["timestamp","is_forecast"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    df.rename(columns={"temperature_2m": "temperature", "wind_speed_10m": "wind_speed", "relative_humidity_2m":"humidity"}, inplace=True)
    print(df.columns)
    df = df.iloc[:-1, :]
    print("final_df: \n", df)
    print(df["timestamp"].diff().value_counts())
    df.to_parquet(out_path)
    print(f"[OK] weather unified: {out_path}")

def fetch_weather():

    # Open-Meteo：1日1回で“実績相当＋7日予報”を取得 → 30分化 → 保存
    lat, lon = 35.6812, 139.7671
    build_openmeteo_daily(
        lat=lat, lon=lon, tz="Asia/Tokyo",
        past_days=7, forecast_days=7
    )
        
if __name__ == "__main__":
    fetch_weather()