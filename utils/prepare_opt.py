"""
- 数理最適化に入力する太陽光および風力の上限
- 燃料のコスト
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pytz

CACHE_DIR = Path("data/cache")

WEATHER_DIR = CACHE_DIR / "weather_bf1w_af1w.parquet"
MARKET_DIR  = CACHE_DIR / "fx_commodity_day.parquet"   # date,USDJPY,brent,henry_hub,coal_aus
DEMAND_DIR = CACHE_DIR / "demand_forecast.parquet"    # date,USDJPY,brent,henry_hub,coal_aus

# 出力ファイル
PV_WIND_OUT = CACHE_DIR / "pv_wind_demand.parquet"
FUEL_OUT = CACHE_DIR / "daily_costs.parquet"

# タイムゾーン
TZ = pytz.timezone("Asia/Tokyo")

# 30分解像度
DT_MIN = 30

# ---- 可調整パラメータ ----
PV_CAP_MW   = 4000.0     # 地域のPV合計容量（MW）
PV_PR       = 0.85       # パフォーマンス比
G_STC       = 1000.0     # W/m^2

WIND_CAP_MW = 1500.0     # 風力合計容量（MW）
HUB_HEIGHT  = 100.0      # ハブ高 (m)
ALPHA       = 0.14       # 風速のべき法則指数（海岸・平野の代表値）

# 風力パワーカーブの代表値（m/s）
V_CUT_IN = 3.0
V_RATED  = 12.0
V_CUTOUT = 25.0

# ---- 可調整パラメータ（熱消費・可変O&M・CO2）----
HEATRATE = {   # MMBtu/MWh
    "lng": 6.5,      # 高効率CCGTの代表
    "coal": 9.0,     # USCの代表
    "oil": 10.0,
}

VOM_YEN_PER_MWH = {   # 可変O&Mの近似（円/MWh）
    "lng": 500.0,
    "coal": 1000.0,
    "oil": 1200.0,
}

CO2_T_PER_MWH = {     # 排出係数（t-CO2/MWh の代表）
    "lng": 0.35,
    "coal": 0.90,
    "oil": 0.75,
}
CO2_PRICE_YEN_PER_T = 3000.0  # 社内カーボンプライスの近似

# ==== 太陽光 上限 ====
def make_pv_avail(weather):

    day = weather["timestamp"].dt.date
    G = weather["shortwave_radiation"].astype(float).clip(lower=0.0)
    G_day_max = G.groupby(day).transform("max").clip(lower=1e-3)
    print("G_day_max: \n", G_day_max[0])
    cf_pv_raw = (G / G_day_max) * PV_PR
    cf_pv = cf_pv_raw.clip(lower=0.0, upper=1.0)

    weather["pv_cf"] = cf_pv
    weather["pv_avail_MW"] = PV_CAP_MW * cf_pv
    
    out_cols = [
        "timestamp",
        "pv_avail_MW",
    ]
    
    weather = weather[out_cols]
    return weather


# ==== 風力 上限 ====
def make_wind_avail(weather):
    
    v10 = weather["wind_speed"].astype(float)
    v_hub = v10 * (HUB_HEIGHT / 10.0) ** ALPHA

    cf_wind = np.zeros(48)
    mask1 = (v_hub >= V_CUT_IN) & (v_hub < V_RATED)
    mask2 = (v_hub >= V_RATED) & (v_hub < V_CUTOUT)

    cf_wind[mask1] = ((v_hub[mask1] - V_CUT_IN) / (V_RATED - V_CUT_IN)) ** 3
    cf_wind[mask2] = 1.0

    weather["wind_cf"] = cf_wind
    weather["wind_avail_MW"] = WIND_CAP_MW * cf_wind

    out_cols = [
        "timestamp",
        "wind_avail_MW",
    ]
    weather = weather[out_cols]
    return weather

# ==== 燃料 コスト ====
def make_fuel_cost(market):

    # 変換：$/bbl → $/MMBtu（原油）
    market["oil_usd_per_mmbtu"]  = market["brent"] / 5.8
    # 変換：$/t → $/MMBtu（一般炭：24 MMBtu/t）
    market["coal_usd_per_mmbtu"] = market["coal_aus"] / 24.0
    # LNGはHenry Hubを簡便に採用（$/MMBtu）
    market["lng_usd_per_mmbtu"]  = market["henry_hub"]
    # 円/MMBtu
    market["oil_jpy_per_mmbtu"]  = market["oil_usd_per_mmbtu"]  * market["USDJPY"]
    market["coal_jpy_per_mmbtu"] = market["coal_usd_per_mmbtu"] * market["USDJPY"]
    market["lng_jpy_per_mmbtu"]  = market["lng_usd_per_mmbtu"]  * market["USDJPY"]
    
    # 円/MWh = 円/MMBtu × HeatRate + VOM + CO2
    def varcost_yen_per_mwh(row, fuel):
        fuel_jpy_per_mmbtu = row[f"{fuel}_jpy_per_mmbtu"] if fuel != "coal" else row["coal_jpy_per_mmbtu"]
        fuel_key = "lng" if fuel=="lng" else ("coal" if fuel=="coal" else "oil")
        return (fuel_jpy_per_mmbtu * HEATRATE[fuel_key]
                + VOM_YEN_PER_MWH[fuel_key]
                + CO2_PRICE_YEN_PER_T * CO2_T_PER_MWH[fuel_key])

    market["c_lng"]  = market.apply(varcost_yen_per_mwh, axis=1, fuel="lng")
    market["c_coal"] = market.apply(varcost_yen_per_mwh, axis=1, fuel="coal")
    market["c_oil"]  = market.apply(varcost_yen_per_mwh, axis=1, fuel="oil")
    
    daily = market[["timestamp","c_coal","c_oil","c_lng"]]  # 必要列だけ
    
    return daily


def prepare_opt(today):

    # today = pd.Timestamp(datetime.today().date())
    # # tomorrow= today + timedelta(days=1)
    weather = pd.read_parquet(WEATHER_DIR)

    weather = weather[weather["timestamp"].dt.date == today] # 対象の日付だけ
    weather.reset_index(drop=True, inplace=True)
    print("weather df: \n", weather)
    
    pv_df = make_pv_avail(weather)
    wind_df = make_wind_avail(weather)

    pv_wind = pd.DataFrame({
        "timestamp": pv_df["timestamp"],
        "pv_avail_MW": pv_df["pv_avail_MW"].astype(float),
        "wind_avail_MW": wind_df["wind_avail_MW"].astype(float),
    })

    pv_wind.to_parquet(PV_WIND_OUT)
    print(f"[SAVE] pv_wind: {PV_WIND_OUT}")
    print("pv_wind: \n", pv_wind)
    
    market = pd.read_parquet(MARKET_DIR)
    market = market[market["timestamp"].dt.date == today] # 対象の日付だけ
    print("market: \n", market)
    
    mkt = make_fuel_cost(market)
    mkt.reset_index(drop=True, inplace=True)
    mkt.to_parquet(FUEL_OUT)
    print(f"[SAVE] mkt_df: {FUEL_OUT}")
    print("mkt: \n", mkt)
    
    return pv_wind, mkt
