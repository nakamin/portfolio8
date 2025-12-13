"""
- 数理最適化に入力する太陽光および風力の上限
- 燃料のコスト
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Literal
import pytz

CACHE_DIR = Path("data/cache")

WEATHER_DIR = CACHE_DIR / "weather_bf1w_af1w.parquet"
MARKET_DIR  = CACHE_DIR / "fx_commodity_day.parquet"   # date,USDJPY,brent,henry_hub,coal_aus

# 出力ファイル
PV_WIND_OUT = CACHE_DIR / "pv_wind_demand.parquet"
FUEL_OUT = CACHE_DIR / "daily_costs.parquet"

# タイムゾーン
TZ = pytz.timezone("Asia/Tokyo")

# 30分解像度
DT_MIN = 30

# ==========
# Parameters (暫定値)
# ==========
PV_CAP_INIT_MW: float = 4000.0   # PV設備容量（MW）: 暫定
PV_PR: float = 0.85         # Performance Ratio（損失を含む係数）
G_STC: float = 1000.0       # STC（標準試験条件）の放射照度 [W/m^2]

WIND_CAP_INIT_MW: float = 1500.0 # 風力設備容量（MW）: 暫定
HUB_HEIGHT: float = 100.0   # ハブ高 [m]
ALPHA: float = 0.14         # べき法則指数（代表値）

# 風力パワーカーブ（代表値）
V_CUT_IN: float = 3.0
V_RATED: float = 12.0
V_CUTOUT: float = 25.0

# 燃料コスト変換用（熱消費・可変O&M・CO2）
HEATRATE = {  # MMBtu/MWh
    "lng": 6.5,
    "coal": 9.0,
    "oil": 10.0,
}
VOM_YEN_PER_MWH = {
    "lng": 500.0,
    "coal": 1000.0,
    "oil": 1200.0,
}
CO2_T_PER_MWH = {
    "lng": 0.35,
    "coal": 0.90,
    "oil": 0.75,
}
CO2_PRICE_YEN_PER_T: float = 3000.0

# ==== 太陽光 上限 ====
def pv_capacity_factor_stc(
    shortwave_radiation_wm2: pd.Series,
    pv_pr: float = PV_PR,
    g_stc: float = G_STC,
) -> pd.Series:
    """
    太陽光の稼働率（0〜1）を、STC（標準試験条件）ベースの簡易式で作る
    """
    G = shortwave_radiation_wm2.astype(float).clip(lower=0.0)
    cf_raw = (G / float(g_stc)) * float(pv_pr)
    return cf_raw.clip(lower=0.0, upper=1.0)

def make_pv_avail(weather, cap_mw: float = PV_CAP_INIT_MW,):

    cf = pv_capacity_factor_stc(weather["shortwave_radiation"], pv_pr=PV_PR, g_stc=G_STC)
    out = pd.DataFrame({
        "timestamp": weather["timestamp"],
        "pv_cf": cf.astype(float),
        "pv_avail_MW": (float(cap_mw) * cf.astype(float)),
    })
    return out[["timestamp", "pv_avail_MW"]]


# ==== 風力 上限 ====
def wind_capacity_factor_power_curve(v_hub: np.ndarray) -> np.ndarray:
    """
    風力の簡易パワーカーブ（稼働率 0〜1）。
    - v < cut-in: 0
    - cut-in <= v < rated: 立ち上がり（ここでは3乗で近似）
    - rated <= v < cut-out: 1
    - v >= cut-out: 0（安全停止）
    """
    cf = np.zeros_like(v_hub, dtype=float)

    mask1 = (v_hub >= V_CUT_IN) & (v_hub < V_RATED)
    mask2 = (v_hub >= V_RATED) & (v_hub < V_CUTOUT)

    # ratedに向かって滑らかに増える近似：((v - cut_in)/(rated - cut_in))^3
    cf[mask1] = ((v_hub[mask1] - V_CUT_IN) / (V_RATED - V_CUT_IN)) ** 3
    cf[mask2] = 1.0
    # cut-out以上は 0 のまま
    return np.clip(cf, 0.0, 1.0)

def make_wind_avail(weather, cap_mw: float = WIND_CAP_INIT_MW,):
    """
    weather から 風力の時刻別上限（wind_avail_MW）を作る
    """
    v10 = weather["wind_speed"].astype(float).to_numpy()
    v_hub = v10 * (HUB_HEIGHT / 10.0) ** ALPHA

    cf = wind_capacity_factor_power_curve(v_hub)

    out = pd.DataFrame({
        "timestamp": weather["timestamp"],
        "wind_cf": cf.astype(float),
        "wind_avail_MW": (float(cap_mw) * cf.astype(float)),
    })
    return out[["timestamp", "wind_avail_MW"]]


# ==== 燃料 コスト ====
def make_fuel_cost(market):
    """
    市場データ（1日分）から、火力の可変費（円/MWh）を作る
    - 1) 商品価格を $/MMBtu に寄せる（単位変換）
    - 2) USDJPY を掛けて 円/MMBtu にする
    - 3) (円/MMBtu × heatrate) + 可変O&M + CO2コスト を足して 円/MWh にする
    変換の前提：
    - 原油: 1 bbl ≒ 5.8 MMBtu
    - 一般炭: 1 t ≒ 24 MMBtu
    - LNG: Henry Hub を $/MMBtu としてそのまま
    """

    # $/MMBtu への変換（簡便）
    market["oil_usd_per_mmbtu"]  = market["brent"].astype(float) / 5.8
    market["coal_usd_per_mmbtu"] = market["coal_aus"].astype(float) / 24.0
    market["lng_usd_per_mmbtu"]  = market["henry_hub"].astype(float)

    # 円/MMBtu（為替掛け）
    fx = market["USDJPY"].astype(float)
    market["oil_jpy_per_mmbtu"]  = market["oil_usd_per_mmbtu"]  * fx
    market["coal_jpy_per_mmbtu"] = market["coal_usd_per_mmbtu"] * fx
    market["lng_jpy_per_mmbtu"]  = market["lng_usd_per_mmbtu"]  * fx

    def varcost_yen_per_mwh(row: pd.Series, fuel: Literal["lng", "coal", "oil"]) -> float:
        fuel_jpy_per_mmbtu = float(row[f"{fuel}_jpy_per_mmbtu"])
        return (
            fuel_jpy_per_mmbtu * float(HEATRATE[fuel])
            + float(VOM_YEN_PER_MWH[fuel])
            + float(CO2_PRICE_YEN_PER_T) * float(CO2_T_PER_MWH[fuel])
        )

    market["c_lng"]  = market.apply(varcost_yen_per_mwh, axis=1, fuel="lng")
    market["c_coal"] = market.apply(varcost_yen_per_mwh, axis=1, fuel="coal")
    market["c_oil"]  = market.apply(varcost_yen_per_mwh, axis=1, fuel="oil")

    return market[["timestamp", "c_coal", "c_oil", "c_lng"]]

# ===== 実績ベースで上限を作成 =====
def estimate_pmax_long_term(series, long_conf, stat):
    """
    過去全期間の実績（30分値）から、「構造的な上限」を推定
    - stat: 
        - "quantile": 日次最大の分布の上側分位点（例：0.995）× margin
        - "avg":      日次最大の平均 × avg_margin（ベースロード系向けに使う想定）
    """

    if series is None or series.empty:
        return np.nan
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series.index must be DatetimeIndex")
    
    daily_max = series.resample("1D").max()
    if daily_max.empty:
        return np.nan

    if stat == "avg":
        mean_val = float(daily_max.mean())
        margin = float(long_conf.get("avg_margin", 1.0))
        return mean_val * margin

    q = float(long_conf.get("quantile", 0.995))
    margin = float(long_conf.get("margin", 1.05))
    ref = float(daily_max.quantile(q))
    return ref * margin
    
def recent_max(series: pd.Series, today: pd.Timestamp, recent_days: int = 7) -> float:
    """
    直近 recent_days 日の 1日最大値のうち最大を返す。
    series: 30分値などの時系列（DatetimeIndex）
    today: 今日（例: 2025-12-13）
    """
    if series is None or series.empty:
        return np.nan
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series.index must be DatetimeIndex")

    start = pd.Timestamp(today) - pd.Timedelta(days=int(recent_days))
    end = pd.Timestamp(today) + pd.Timedelta(days=1)

    s = series.loc[(series.index >= start) & (series.index < end)]
    if s.empty:
        return np.nan

    daily_max = s.resample("1D").max()
    if daily_max.empty:
        return np.nan

    return float(daily_max.max())

def effective_pmax(series, long_conf, short_conf, today, stat, fallback_prev_cap=None, fallback_init_cap=None):
    """
    - 長期 + 直近を組み合わせた「今日の P_max」を計算
    - base = 長期構造的上限
    - recent_cap = 直近の最大出力 * alpha
    - → P_max_today = min(base, recent_cap)
    """
    base = estimate_pmax_long_term(
        series,
        long_conf=long_conf,
        stat=stat
    )
    print("base: \n", base)
    
    recent_days = int(short_conf.get("recent_days", 7))
    rmax = recent_max(series, today, recent_days=recent_days)
    print("rmax: \n", rmax)
    if rmax is None:
        return base
    recent_cap = rmax * float(short_conf.get("alpha", 1.05))
    
    # recent_cap が使えない場合（欠損 or ほぼ0）
    min_valid = float(short_conf.get("min_valid", 1e-6))
    if pd.isna(recent_cap) or recent_cap <= min_valid:
        if fallback_prev_cap is not None and fallback_prev_cap > min_valid:
            return float(fallback_prev_cap)
        if not pd.isna(base) and base > min_valid:
            return float(base)
        if fallback_init_cap is not None and fallback_init_cap > min_valid:
            return float(fallback_init_cap)
        return 0.0

    # base が欠損なら recent を採用
    if pd.isna(base) or base <= min_valid:
        return float(recent_cap)

    return float(min(base, recent_cap))

def prepare_opt(
    today,
    pv_cap_mw: float = PV_CAP_INIT_MW,
    wind_cap_mw: float = WIND_CAP_INIT_MW,
    ):

    # today = pd.Timestamp(datetime.today().date())
    # # tomorrow= today + timedelta(days=1)
    weather = pd.read_parquet(WEATHER_DIR)

    weather = weather[weather["timestamp"].dt.date == today] # 対象の日付だけ
    weather.reset_index(drop=True, inplace=True)
    print("weather df: \n", weather)
    
    pv_df = make_pv_avail(weather, cap_mw=pv_cap_mw)
    wind_df = make_wind_avail(weather, cap_mw=wind_cap_mw)

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