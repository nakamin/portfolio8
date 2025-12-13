from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from copy import deepcopy
from typing import Dict, Any, Optional
import pytz
from datetime import datetime, timedelta, time, date

from pyomo.environ import (
    ConcreteModel, Var, NonNegativeReals, Reals, Set, Param,
    Objective, Constraint, ConstraintList, minimize, value, SolverFactory
)

from utils.prepare_opt import prepare_opt, effective_pmax

CACHE_DIR = Path("data/cache")
DATA_DIR = Path("data")

ACTUAL_DEMAND_PATH = DATA_DIR / "actual.parquet"
DEMAND_PATH = CACHE_DIR / "demand_forecast.parquet"
WEATHER_PATH = CACHE_DIR / "weather_bf1w_af1w.parquet"
CONFIG_PATH = Path("config/params.yaml")
CONFIG_RESOLVED_PATH = Path("config/params_resolved.yaml")

OUT_PATH = CACHE_DIR / "dispatch_optimal.parquet"

# ==========
# YAML params
# ==========
def load_params(path="params.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_params(params, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)

# ==========
# Capacity estimation from actuals
# ==========
def apply_pmax_from_actuals(params_raw, long_df, demand_forecast_df, today=None):
    """
    - params.yaml を読み込み、昨日までの実績データからP_maxを推定し、mode:estimateに設定された電源に反映する
    """
    params = deepcopy(params_raw) # コピーして編集用に使う
    cap_conf = params.get("capacity", {})
    long_conf = cap_conf.get("long_term", {})
    short_conf = cap_conf.get("short_term", {})
    sources_conf = cap_conf.get("sources", {})
    print("sources_conf", sources_conf)

    target_cols = [
        "lng", "coal", "oil", "th_other",
        "hydro", "biomass", "misc",
        "pstorage", "battery", "tie",
    ]
    pmax_est = {}

    # 各内部名 g について P_max を計算
    for col in target_cols:
        mode = sources_conf.get(col, {}).get("mode", "fixed")
        if mode != "estimate":
            continue

        clip_pos = col in ("pstorage", "battery", "tie")
        series = _history_series(
            long_df=long_df,
            recent_df=demand_forecast_df,
            col=col,
            today=today,
            clip_positive=clip_pos,
        )
        if series is None or series.empty:
            continue

        stat = sources_conf.get(col, {}).get("stat", "quantile")

        pmax_est[col] = effective_pmax(
            series,
            long_conf=long_conf,
            short_conf=short_conf,
            today=pd.Timestamp(today),
            stat=stat,
        )
        
    # thermal
    if "thermal" in params:
        for g in params["thermal"].keys():
            if g in pmax_est:
                params["thermal"][g]["P_max"] = float(round(pmax_est[g], 0))

    # hydro
    if "hydro" in pmax_est and "hydro" in params:
        params["hydro"]["P_max"] = float(round(pmax_est["hydro"], 0))

    # pstorage, battery, tie
    if "pstorage" in pmax_est and "pstorage" in params:
        params["pstorage"]["P_gen_max"] = float(round(pmax_est["pstorage"], 0))
    if "battery" in pmax_est and "battery" in params:
        params["battery"]["P_dis_max"] = float(round(pmax_est["battery"], 0))
    if "tie" in pmax_est and "tie" in params:
        params["tie"]["cap"] = float(round(pmax_est["tie"], 0))

    # ログ用に埋め込んでおく（オプション）
    params.setdefault("capacity", {})
    params["capacity"]["pmax_today"] = pmax_est

    return params

# ==========
# Costs
# ==========
def build_costs(params: dict, fuels_row) -> dict:
    """
    thermal セクションと fuels_row から costs dict を作る
    - coal, oil, lng: fuels_row の c_* で上書き
    - それ以外(th_other, biomass, miscなど): params.yaml の c を使う
    """
    thermal_conf = params["thermal"]

    fuel_col_map = {"coal": "c_coal", "oil": "c_oil", "lng": "c_lng"}

    costs: Dict[str, float] = {}
    for g, conf in thermal_conf.items():
        col = fuel_col_map.get(g)
        if col is not None and col in fuels_row.index:
            costs[g] = float(fuels_row[col])
        else:
            costs[g] = float(conf["c"])

    return costs

# ==========
# Optimization
# ==========
def solve_dispatch(df_ts, costs, params, init_state):
    """
    df_ts: DataFrame[timestamp, load_MW, pv_avail_MW, wind_avail_MW]
    costs: {"coal": 円/MWh, "oil": 円/MWh, "lng": 円/MWh}
    params: params_today（P_max埋め込み済み）
    init_state: 直前時刻の出力（簡略に使う）。
    """
    dt = float(params.get("dt_hours", 0.5))
    rho = float(params["reserve"]["rho"])
    c_curt_pv = float(params["penalty"]["curtail_pv"])
    c_curt_wind = float(params["penalty"]["curtail_wind"])
    c_shed = float(params["penalty"]["shed"])

    # 時間インデックス
    df = df_ts.copy().reset_index(drop=True)
    T = list(df.index)

    m = ConcreteModel()
    m.T = Set(initialize=T)

    # パラメータ（負荷と再エネ上限）
    load  = {t: float(df.loc[t, "predicted_demand"])       for t in T}
    pv_av = {t: float(df.loc[t, "pv_avail_MW"])   for t in T}
    w_av  = {t: float(df.loc[t, "wind_avail_MW"]) for t in T}

    # P_max & ramp を params から取得
    Pmax_hy = float(params["hydro"]["P_max"])
    RU_hy = float(params["hydro"]["RU"])
    RD_hy = float(params["hydro"]["RD"])
    E_day_hy = float(params["hydro"]["E_day"])

    thermal_keys = ["coal", "oil", "lng", "th_other", "biomass", "misc"]
    Pmax_th = {g: float(params["thermal"][g]["P_max"]) for g in thermal_keys}
    RU_th = {g: float(params["thermal"][g]["RU"]) for g in thermal_keys}
    RD_th = {g: float(params["thermal"][g]["RD"]) for g in thermal_keys}

    Cap_tie = float(params["tie"]["cap"])
    c_import = float(params["tie"]["c_import"])

    bat_conf = params["battery"]
    ps_conf = params["pstorage"]

    # 変数
    m.P_pv   = Var(m.T, domain=NonNegativeReals)
    m.P_wind = Var(m.T, domain=NonNegativeReals)
    m.Curt_pv   = Var(m.T, domain=NonNegativeReals)
    m.Curt_wind = Var(m.T, domain=NonNegativeReals)

    m.P_hy   = Var(m.T, domain=NonNegativeReals)
    m.P_coal = Var(m.T, domain=NonNegativeReals)
    m.P_oil  = Var(m.T, domain=NonNegativeReals)
    m.P_lng  = Var(m.T, domain=NonNegativeReals)
    m.P_th_other  = Var(m.T, domain=NonNegativeReals)
    m.P_biomass = Var(m.T, domain=NonNegativeReals)
    m.P_misc   = Var(m.T, domain=NonNegativeReals)

    m.R_hy   = Var(m.T, domain=NonNegativeReals)
    m.R_coal = Var(m.T, domain=NonNegativeReals)
    m.R_oil  = Var(m.T, domain=NonNegativeReals)
    m.R_lng  = Var(m.T, domain=NonNegativeReals)
    m.R_th_other  = Var(m.T, domain=NonNegativeReals)
    m.R_biomass = Var(m.T, domain=NonNegativeReals)
    m.R_misc   = Var(m.T, domain=NonNegativeReals)

    m.Shed   = Var(m.T, domain=NonNegativeReals)  # 未供給（停電）

    # 連系線（インポート）
    m.P_imp  = Var(m.T, domain=NonNegativeReals)  # 流入
    m.R_imp  = Var(m.T, domain=NonNegativeReals)  # 予備力としての余力

    # 蓄電池
    m.P_ch   = Var(m.T, domain=NonNegativeReals)  # 充電
    m.P_dis  = Var(m.T, domain=NonNegativeReals)  # 放電
    m.E_bat  = Var(m.T, domain=NonNegativeReals)  # SoC
    m.R_bat  = Var(m.T, domain=NonNegativeReals)  # 上方予備

    # 揚水
    m.P_pump = Var(m.T, domain=NonNegativeReals)  # 揚水（消費）
    m.P_gen  = Var(m.T, domain=NonNegativeReals)  # 発電
    m.E_ps   = Var(m.T, domain=NonNegativeReals)
    m.R_ps   = Var(m.T, domain=NonNegativeReals)

    # 目的関数：燃料費 + 抑制ペナルティ + インポートコスト + 停電ペナルティ
    def obj_rule(_):
        return sum(
            dt * (
                costs["coal"] * m.P_coal[t]
                + costs["oil"] * m.P_oil[t]
                + costs["lng"] * m.P_lng[t]
                + costs["th_other"] * m.P_th_other[t]
                + costs["biomass"] * m.P_biomass[t]
                + costs["misc"] * m.P_misc[t]
                + c_curt_pv * m.Curt_pv[t]
                + c_curt_wind * m.Curt_wind[t]
                + c_import * m.P_imp[t]
                + c_shed * m.Shed[t]
            )
            for t in T
        )

    m.OBJ = Objective(rule=obj_rule, sense=minimize)

    # 可用性（再エネ上限と抑制）
    m.PV_av = Param(m.T, initialize=pv_av, within=Reals)
    m.W_av = Param(m.T, initialize=w_av, within=Reals)

    def pv_rule(_, t):
        return m.P_pv[t] + m.Curt_pv[t] == m.PV_av[t]

    def wind_rule(_, t):
        return m.P_wind[t] + m.Curt_wind[t] == m.W_av[t]

    m.PVAvail = Constraint(m.T, rule=pv_rule)
    m.WindAvail = Constraint(m.T, rule=wind_rule)

    # 出力上限（P_max - 予備力）
    m.CapHy = Constraint(m.T, rule=lambda _, t: m.P_hy[t] <= Pmax_hy - m.R_hy[t])
    m.CapCoal = Constraint(m.T, rule=lambda _, t: m.P_coal[t] <= Pmax_th["coal"] - m.R_coal[t])
    m.CapOil = Constraint(m.T, rule=lambda _, t: m.P_oil[t] <= Pmax_th["oil"] - m.R_oil[t])
    m.CapLng = Constraint(m.T, rule=lambda _, t: m.P_lng[t] <= Pmax_th["lng"] - m.R_lng[t])
    m.CapOth = Constraint(m.T, rule=lambda _, t: m.P_th_other[t] <= Pmax_th["th_other"] - m.R_th_other[t])
    m.CapBio = Constraint(m.T, rule=lambda _, t: m.P_biomass[t] <= Pmax_th["biomass"] - m.R_biomass[t])
    m.CapMis = Constraint(m.T, rule=lambda _, t: m.P_misc[t] <= Pmax_th["misc"] - m.R_misc[t])

    # 連系線容量
    m.CapTie = Constraint(m.T, rule=lambda _, t: m.P_imp[t] + m.R_imp[t] <= Cap_tie)

    # 予備力制約（上方予備）
    def reserve_rule(_, t):
        return (
            m.R_hy[t]
            + m.R_coal[t] + m.R_oil[t] + m.R_lng[t]
            + m.R_th_other[t] + m.R_biomass[t] + m.R_misc[t]
            + m.R_bat[t] + m.R_ps[t] + m.R_imp[t]
        ) >= rho * load[t]

    m.Reserve = Constraint(m.T, rule=reserve_rule)

    # 蓄電池のエネルギーバランス
    eta_ch = float(bat_conf["eta_ch"])
    eta_dis = float(bat_conf["eta_dis"])
    Emax_b = float(bat_conf["E_max"])
    Pchmax = float(bat_conf["P_ch_max"])
    Pdismax = float(bat_conf["P_dis_max"])
    E0_b = float(bat_conf["E0"])

    m.BatSOC = ConstraintList()
    m.BatCap = ConstraintList()

    # t=0
    m.BatSOC.add(m.E_bat[0] == E0_b + eta_ch*m.P_ch[0]*dt - (1/eta_dis)*m.P_dis[0]*dt)
    m.BatCap.add(m.E_bat[0] <= Emax_b)
    m.BatCap.add(m.P_ch[0]  <= Pchmax)
    m.BatCap.add(m.P_dis[0] <= Pdismax)
    # t>=1
    for t in T[1:]:
        m.BatSOC.add(m.E_bat[t] == m.E_bat[t-1] + eta_ch*m.P_ch[t]*dt - (1/eta_dis)*m.P_dis[t]*dt)
        m.BatCap.add(m.E_bat[t] <= Emax_b)
        m.BatCap.add(m.P_ch[t]  <= Pchmax)
        m.BatCap.add(m.P_dis[t] <= Pdismax)

    # バッテリー予備力（上方：これから放電できる余力）
    m.BatRes = ConstraintList()
    for t in T:
        m.BatRes.add(m.R_bat[t] <= Pdismax - m.P_dis[t])
        m.BatRes.add(m.R_bat[t] <= m.E_bat[t] / dt)

    # 揚水のエネルギーバランス
    eta_pump = float(ps_conf["eta_pump"])
    eta_gen = float(ps_conf["eta_gen"])
    Emax_ps = float(ps_conf["E_max"])
    Ppumpmax = float(ps_conf["P_pump_max"])
    Pgenmax = float(ps_conf["P_gen_max"])
    E0_ps = float(ps_conf["E0"])

    m.PSSOC = ConstraintList()
    m.PSCap = ConstraintList()
    
    m.PSSOC.add(m.E_ps[0] == E0_ps + eta_pump*m.P_pump[0]*dt - (1/eta_gen)*m.P_gen[0]*dt)
    m.PSCap.add(m.E_ps[0] <= Emax_ps)
    m.PSCap.add(m.P_pump[0] <= Ppumpmax)
    m.PSCap.add(m.P_gen[0]  <= Pgenmax)
    for t in T[1:]:
        m.PSSOC.add(m.E_ps[t] == m.E_ps[t-1] + eta_pump*m.P_pump[t]*dt - (1/eta_gen)*m.P_gen[t]*dt)
        m.PSCap.add(m.E_ps[t] <= Emax_ps)
        m.PSCap.add(m.P_pump[t] <= Ppumpmax)
        m.PSCap.add(m.P_gen[t]  <= Pgenmax)

    # 揚水予備力
    m.PSRes = ConstraintList()
    for t in T:
        m.PSRes.add(m.R_ps[t] <= Pgenmax - m.P_gen[t])
        m.PSRes.add(m.R_ps[t] <= m.E_ps[t] / dt)

    # ランプ制約（火力と水力）
    init = init_state or {k: 0.0 for k in ["hy"] + thermal_keys}

    def _get_init(key: str) -> float:
        return float(init.get(key, 0.0))

    m.Ramp = ConstraintList()
    # t=0
    m.Ramp.add(m.P_hy[0] - _get_init("hy") <= RU_hy)
    m.Ramp.add(_get_init("hy") - m.P_hy[0] <= RD_hy)
    for g in thermal_keys:
        Pg = getattr(m, f"P_{g}")
        m.Ramp.add(Pg[0] - _get_init(g) <= RU_th[g])
        m.Ramp.add(_get_init(g) - Pg[0] <= RD_th[g])

    # t>=1
    for t in T[1:]:
        m.Ramp.add(m.P_hy[t] - m.P_hy[t - 1] <= RU_hy)
        m.Ramp.add(m.P_hy[t - 1] - m.P_hy[t] <= RD_hy)
        for g in thermal_keys:
            Pg = getattr(m, f"P_{g}")
            m.Ramp.add(Pg[t] - Pg[t - 1] <= RU_th[g])
            m.Ramp.add(Pg[t - 1] - Pg[t] <= RD_th[g])

    # 水力（通常水力）の日次エネルギー上限
    m.HydroDay = ConstraintList()
    df["date"] = df["timestamp"].dt.date
    for d, idx in df.groupby("date").groups.items():
        m.HydroDay.add(sum(m.P_hy[t]*dt for t in idx) <= E_day_hy)

    # 需給バランス
    # 左辺：各電源の実出力 + インポート + 揚水発電 + 蓄電池放電
    # 右辺：需要 + 揚水の揚水 + 蓄電池充電 + 未供給
    def balance_rule(_, t):
        return (
            m.P_pv[t] + m.P_wind[t] + m.P_hy[t]
            + m.P_coal[t] + m.P_oil[t] + m.P_lng[t]
            + m.P_th_other[t] + m.P_biomass[t] + m.P_misc[t]
            + m.P_gen[t] + m.P_dis[t] + m.P_imp[t]
        ) == (
            load[t] + m.P_pump[t] + m.P_ch[t] + m.Shed[t]
        )

    m.Balance = Constraint(m.T, rule=balance_rule)

    solver = SolverFactory("highs")
    solver.solve(m, tee=False)

    # 結果DataFrame
    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "pv":    [value(m.P_pv[t])   for t in T],
        "wind":  [value(m.P_wind[t]) for t in T],
        "hydro": [value(m.P_hy[t])   for t in T],
        "coal":  [value(m.P_coal[t]) for t in T],
        "oil":   [value(m.P_oil[t])  for t in T],
        "lng":   [value(m.P_lng[t])  for t in T],
        "th_other":   [value(m.P_th_other[t])  for t in T],
        "biomass": [value(m.P_biomass[t]) for t in T],
        "misc":   [value(m.P_misc[t])   for t in T],
        "pstorage_gen": [value(m.P_gen[t])  for t in T],
        "pstorage_pump":[value(m.P_pump[t]) for t in T],
        "battery_dis":  [value(m.P_dis[t])  for t in T],
        "battery_ch":   [value(m.P_ch[t])   for t in T],
        "import":       [value(m.P_imp[t])  for t in T],
        "curtail_pv":   [value(m.Curt_pv[t])   for t in T],
        "curtail_wind": [value(m.Curt_wind[t]) for t in T],
        "reserve": [
            value(m.R_hy[t] + m.R_coal[t] + m.R_oil[t] + m.R_lng[t]
                  + m.R_bat[t] + m.R_ps[t] + m.R_imp[t])
            for t in T
        ],
        "shed":    [value(m.Shed[t])    for t in T],
        "predicted_demand":[load[t] for t in T],
    })
    out["total_cost"] = (
        out["coal"]*costs["coal"] +
        out["oil"]*costs["oil"] +
        out["lng"]*costs["lng"] +
        out["th_other"]*costs["th_other"] +
        out["biomass"]*costs["biomass"] +
        out["misc"]*costs["misc"] +
        out["curtail_pv"]*c_curt_pv +
        out["curtail_wind"]*c_curt_wind +
        out["shed"]*c_shed +
        out["import"]*c_import
    )
    return out

def _history_series(
    long_df: Optional[pd.DataFrame],
    recent_df: Optional[pd.DataFrame],
    col: str,
    *,
    today: date,
    clip_positive: bool = False,
) -> Optional[pd.Series]:
    """
    上限制約推定用の履歴系列を作る

    - long_df: 長期構造（例: 2024/02〜前月など）※列が無い場合は無視
    - recent_df: 直近（例: demand_forecast。昨日の実績を含む想定）※列が無い場合は無視
    - インデックスは DatetimeIndex 前提
    - 同一timestampが重複した場合は recent_df を優先
    """
    parts = []
    for df in (long_df, recent_df):
        if df is None:
            continue
        if col not in df.columns:
            continue
        s = pd.Series(df[col].astype(float).values, index=df.index)
        parts.append(s)

    if not parts:
        return None

    s_all = pd.concat(parts).sort_index()
    # 重複timestampは最後（= recent側）を採用
    s_all = s_all[~s_all.index.duplicated(keep="last")]

    # 「今日」以降は使わない
    today_ts = pd.Timestamp(today)
    s_all = s_all.loc[s_all.index < today_ts + pd.Timedelta(days=1)]

    if clip_positive:
        s_all = s_all.clip(lower=0.0)
    return s_all


def optimize_dispatch():

    tz = pytz.timezone("Asia/Tokyo")
    today = datetime.now(tz).date()
    # today = datetime.now(tz).date() - timedelta(days=1)
    params_raw = load_params(CONFIG_PATH)
    actual = pd.read_parquet(ACTUAL_DEMAND_PATH) # 2024年2月～前月まで
    print("actual: ", actual)

    demand = pd.read_parquet(DEMAND_PATH) # 予測値が入っている
    demand.set_index("timestamp", inplace=True)
    print("demand: ", demand)
    
    params_today = apply_pmax_from_actuals(params_raw, actual, demand, today)
    print("params_today: ", params_today)
    save_params(params_today, CONFIG_RESOLVED_PATH)

    cap_conf = params_raw.get("capacity", {})
    long_conf = cap_conf.get("long_term", {})
    short_conf = cap_conf.get("short_term", {})

    pv_series = _history_series(actual, demand, "pv", today=today, clip_positive=True)
    wind_series = _history_series(actual, demand, "wind", today=today, clip_positive=True)

    # fallback: params.yaml の暫定値を使う
    pv_cap_fallback = float(params_raw.get("pv_cap_mw_init", 4000.0))
    wind_cap_fallback = float(params_raw.get("wind_cap_mw_init", 1500.0))

    pv_cap_today = pv_cap_fallback
    if pv_series is not None and not pv_series.empty:
        pv_cap_today = float(effective_pmax(
            pv_series, long_conf=long_conf, short_conf=short_conf,
            today=pd.Timestamp(today), stat="quantile"
        )) or pv_cap_fallback

    wind_cap_today = wind_cap_fallback
    if wind_series is not None and not wind_series.empty:
        wind_cap_today = float(effective_pmax(
            wind_series, long_conf=long_conf, short_conf=short_conf,
            today=pd.Timestamp(today), stat="quantile"
        )) or wind_cap_fallback

    # --- 外生データ（pv/wind avail, fuel costs）作成 ---
    pv_wind, fuel_df = prepare_opt(
        today,
        pv_cap_mw=pv_cap_today,
        wind_cap_mw=wind_cap_today,
        )
    fuels_row = fuel_df.iloc[0]  # 1日1行想定
    costs = build_costs(params_today, fuels_row)

    # --- 時系列を結合（最適化で使う） ---
    df_ts = pv_wind.merge(demand, how="left", left_on="timestamp", right_index=True)
    
    yesterday = today - timedelta(days=1)
    yesterday_2330 = pd.Timestamp(pd.Timestamp(datetime.combine(yesterday, time(23, 30))))

    prev_row = demand.loc[[yesterday_2330]]

    init_state = {
        "hy":   float(prev_row["hydro"]),
        "coal": float(prev_row["coal"]),
        "oil":  float(prev_row["oil"]),
        "lng":  float(prev_row["lng"]),
        "th_other":  float(prev_row["th_other"]),
        "biomass":  float(prev_row["biomass"]),
        "misc":  float(prev_row["misc"]),
    }
    
    df_ts = pd.merge(pv_wind, demand, how="left", left_on="timestamp", right_index=True)
    # print("df_ts: \n", df_ts)
    out = solve_dispatch(df_ts, costs, params_today, init_state)

    print("out: \n", out)    
    out.to_parquet(OUT_PATH)
    print(f"[SAVE] dispatch plan to {OUT_PATH}")


if __name__ == "__main__":
    optimize_dispatch()
