from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

import lightgbm as lgb
from typing import Optional, Any
from huggingface_hub import hf_hub_download

from utils.make_date_features import make_date, make_temperature_abs, make_daypart, make_season

CACHE_DIR = Path("data/cache")
HF_REPO_ID = "nakamichan/power-forecast-models"

# 実績と予報
DEMAND_PATH = CACHE_DIR / "demand_forecast.parquet" # 実績も入っている
WEATHER_PATH = CACHE_DIR / "weather_bf1w_af1w.parquet"
MARKET_PATH = CACHE_DIR / "fx_commodity_30min_af1w.parquet"
PRICE_PATH = CACHE_DIR / "spot_tokyo_bf1w_tdy.parquet"
PRICE_EVAL_DETAIl_PATH = CACHE_DIR / "price_evaluation_detail.parquet"
PRICE_HISTORY_PATH = CACHE_DIR / "price_forecast_history.parquet"

# モデル
GAM_PATH = "gam_price.pkl"
LGB10_PATH = "lgb_price_p10.txt"
LGB50_PATH = "lgb_price_p50.txt"
LGB90_PATH = "lgb_price_p90.txt"

LGB10_1w_PATH = "lgb_price_p10_1w.txt"
LGB50_1w_PATH = "lgb_price_p50_1w.txt"
LGB90_1w_PATH = "lgb_price_p90_1w.txt"

# このモデルの結果
PRICE_OUT_PATH = CACHE_DIR / "price_forecast.parquet"
PRICE_EVAL_OUT_PATH = CACHE_DIR / "price_evaluation.parquet"

GAM_FEATURES = [
    # 需要・気象
    "demand", "demand_7d_ma",
    "temperature", "temp_7d_ma",
    "shortwave_radiation",
    "wind_speed",
    # 時間・カレンダー
    "hour", "day_of_year",
    "is_holiday"
    ]

LGB_FEATURES = [
    # 需要・気象関連
    'demand', 'temperature', 'wind_speed',
    'sunshine_duration', 'shortwave_radiation',
    'temperature_abs', 'demand_7d_ma', 'temp_7d_ma',

    # カレンダー・時間関連
    'is_holiday', 'hour',
    'day_of_year', 'season', 'daypart',

    # 為替・燃料価格
    'USDJPY', 'brent', 'henry_hub', 'coal_aus',

    # 為替・燃料価格（移動平均・変化率）
    'USDJPY_ma30', 'USDJPY_chg30',
    'brent_ma30', 'brent_chg30',
    'henry_hub_ma30',
    'coal_aus_ma90', 'coal_aus_chg90',

    # 価格ラグ
    'price_lag_24h', 'price_lag_48h',
    'price_lag_72h', 'price_lag_1w',
    
    # GAM特徴量
    'y_gam'
]

LGB_DROP_FEATURES = [
    # 価格ラグ
    'price_lag_24h', 'price_lag_48h',
    'price_lag_72h',
]

def load_price_model() -> Any:
    """
    価格予測モデルを読み込む
    """
    # モデルファイルを Hub からダウンロード
    gam_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=GAM_PATH,
    )
    lgb10_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=LGB10_PATH,
    )
    lgb50_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=LGB50_PATH,
    )
    lgb90_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=LGB90_PATH,
    )

    lgb10_1w_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=LGB10_1w_PATH,
    )
    lgb50_1w_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=LGB50_1w_PATH,
    )
    lgb90_1w_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=LGB90_1w_PATH,
    )

    gam = joblib.load(gam_path)

    lgb_p10 = lgb.Booster(model_file=lgb10_path)
    lgb_p50 = lgb.Booster(model_file=lgb50_path)
    lgb_p90 = lgb.Booster(model_file=lgb90_path)
    lgb_p10_1w = lgb.Booster(model_file=lgb10_1w_path)
    lgb_p50_1w = lgb.Booster(model_file=lgb50_1w_path)
    lgb_p90_1w = lgb.Booster(model_file=lgb90_1w_path)
    
    return gam, lgb_p10, lgb_p50, lgb_p90, lgb_p10_1w, lgb_p50_1w, lgb_p90_1w

def build_price_features(
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    market: pd.DataFrame,
    price: pd.DataFrame
) -> pd.DataFrame:
    """
    - 特徴量を作成する
    """
    # timestamp を必ず datetime にしておく
    demand["timestamp"] = pd.to_datetime(demand["timestamp"])
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    market["timestamp"] = pd.to_datetime(market["timestamp"])
    price["timestamp"] = pd.to_datetime(price["timestamp"])

    # 結合
    merged_df = market.merge(weather, on="timestamp", how="left") \
                    .merge(demand, on="timestamp", how="left") \
                    .merge(price, on="timestamp", how="left")
    print("merged_df:\n", merged_df)
    
    merged_df = make_date(merged_df)
    
    merged_df['daypart'] = merged_df['hour'].apply(make_daypart)
    merged_df['season']  = merged_df['month'].apply(make_season)
    
    daypart_order = ["dawn","morning","noon","afternoon","evening","night"]
    season_order  = ["spring","summer","autumn","winter"]
    
    merged_df["season"] = pd.Categorical(
    merged_df["season"],
    categories=season_order,
    ordered=False
    )   
    merged_df["daypart"] = pd.Categorical(
        merged_df["daypart"],
        categories=daypart_order,
        ordered=False
    )
    merged_df["is_holiday"] = pd.Categorical(
        merged_df["is_holiday"],
        ordered=False
    )
    
    merged_df["day_of_year"] = merged_df["timestamp"].dt.dayofyear
    # 過去7日移動平均の需要
    merged_df["demand_7d_ma"] = merged_df["demand"].rolling(48*7, min_periods=1).mean()
    # 過去7日移動平均の気温
    merged_df["temp_7d_ma"] = merged_df["temperature"].rolling(48*7, min_periods=1).mean()
    merged_df = make_temperature_abs(merged_df)
    
    # 価格ラグ
    merged_df["price_lag_24h"] = merged_df["tokyo_price_jpy_per_kwh"].shift(48)
    merged_df["price_lag_48h"] = merged_df["tokyo_price_jpy_per_kwh"].shift(96)
    merged_df["price_lag_72h"] = merged_df["tokyo_price_jpy_per_kwh"].shift(144)
    merged_df["price_lag_1w"] = merged_df["tokyo_price_jpy_per_kwh"].shift(336)

    # 為替・石油・ガス（日次レベル）
    for col in ["USDJPY", "brent", "henry_hub"]:
        # 30日移動平均
        merged_df[f"{col}_ma30"] = merged_df[col].rolling(48*30, min_periods=1).mean()
        # 30日前との差分
        merged_df[f"{col}_chg30"] = merged_df[col] - merged_df[col].shift(48*30)

    # 石炭（月次レベル）
    # 90日移動平均
    merged_df["coal_aus_ma90"] = merged_df["coal_aus"].rolling(48*90, min_periods=1).mean()
    # 90日前との差分
    merged_df["coal_aus_chg90"] = merged_df["coal_aus"] - merged_df["coal_aus"].shift(48*90)
    
    return merged_df

def predict_gam(
    gam_model: Optional[Any],
    test_df
) -> pd.DataFrame:
    """
    GAMモデルで推論し、y_gamをtest_dfに代入してLightGBMでも使用
    """
    
    X_gam_test = test_df.copy()
    X_gam_test["is_holiday"] = X_gam_test["is_holiday"].cat.codes.astype("int8")
    X_gam_test = X_gam_test[GAM_FEATURES].to_numpy()
    print("X_gam_test: \n", X_gam_test)
    test_df["y_gam"] = gam_model.predict(X_gam_test)

    return test_df

def predict_lgb(
    lgb_p10, lgb_p50, lgb_p90, X
) -> pd.DataFrame:
    
    print("NaN rate per col:\n", X.isna().mean().sort_values(ascending=False).head(20))

    r10_test = lgb_p10.predict(X)
    r50_test = lgb_p50.predict(X)
    r90_test = lgb_p90.predict(X)

    return r10_test, r50_test, r90_test


def calculate_crps(y_true, quantiles, preds):
    # quantiles: [0.1, 0.5, 0.9]
    # preds: [p10, p50, p90]

    loss = 0
    for q, y_pred in zip(quantiles, preds):
        if y_true < y_pred:
            loss += (1 - q) * (y_pred - y_true)
        else:
            loss += q * (y_true - y_pred)
    return loss

def row_crps(row, quantiles=[0.1, 0.5, 0.9]):
    y_true = float(row["tokyo_price_jpy_per_kwh"])
    p10 = float(row["predicted_price(10%)"])
    p50 = float(row["predicted_price(50%)"])
    p90 = float(row["predicted_price(90%)"])
    
    return float(calculate_crps(
        y_true,
        quantiles,
        [p10, p50, p90]
    ))

def predict_price():
    
    # 特徴量の読み込み
    demand = pd.read_parquet(DEMAND_PATH)
    print("demand: \n", demand)
    weather = pd.read_parquet(WEATHER_PATH)
    print("weather: \n", weather)
    print(weather.columns)
    market = pd.read_parquet(MARKET_PATH)
    print("market: \n", market)
    price = pd.read_parquet(PRICE_PATH)
    print("price: \n", price)
    
    JST = timezone(timedelta(hours=9))
    today = pd.Timestamp(datetime.now(JST).date())

    features_df = build_price_features(demand, weather, market, price)
    print("features_df: \n", features_df)
    
    test_df = features_df[
        (features_df["timestamp"] >= pd.to_datetime(today + timedelta(days=1))) &
        (features_df["timestamp"] < pd.to_datetime(today + timedelta(days=8)))
    ].reset_index(drop=True)
    print("test_df: \n", test_df)
    
    # pred_time = test_df["timestamp"].copy()
    
    gam, lgb_p10, lgb_p50, lgb_p90, lgb_p10_1w, lgb_p50_1w, lgb_p90_1w = load_price_model()

    missing = [c for c in GAM_FEATURES if c not in test_df.columns]
    print("missing GAM cols:", missing)
    
    # GAM
    test_df = predict_gam(gam, test_df)

    missing = [c for c in LGB_FEATURES if c not in test_df.columns]
    print("missing LightGBM cols:", missing)

    test_df_tmw = test_df[
        (test_df["timestamp"] >= pd.to_datetime(today + timedelta(days=1))) &
        (test_df["timestamp"] < pd.to_datetime(today + timedelta(days=2)))
    ].reset_index(drop=True)

    test_df_1w = test_df[
        (test_df["timestamp"] >= pd.to_datetime(today + timedelta(days=2))) &
        (test_df["timestamp"] < pd.to_datetime(today + timedelta(days=8)))
    ].reset_index(drop=True)

    X_tmw = test_df_tmw[LGB_FEATURES]
    X_1w = test_df_1w[[c for c in LGB_FEATURES if c not in LGB_DROP_FEATURES]]
    
    print("test_df_tmw: \n", test_df_tmw)
    print("test_df_1w: \n", test_df_1w)
    
    # LightGBM
    r10_test, r50_test, r90_test = predict_lgb(lgb_p10, lgb_p50, lgb_p90, X_tmw)
    r10_test_1w, r50_test_1w, r90_test_1w = predict_lgb(lgb_p10_1w, lgb_p50_1w, lgb_p90_1w, X_1w)

    # 予測結果をparquetで保存
    pred_tmw = pd.DataFrame({
    "timestamp": test_df_tmw["timestamp"].values,
    "p10": r10_test,
    "p50": r50_test,
    "p90": r90_test,
    })
    pred_1w = pd.DataFrame({
    "timestamp": test_df_1w["timestamp"].values,
    "p10": r10_test_1w,
    "p50": r50_test_1w,
    "p90": r90_test_1w,
    })
    pred_df = pd.concat([pred_tmw, pred_1w], axis=0).sort_values("timestamp").reset_index(drop=True)

    p10 = pred_df["p10"].values
    p50 = pred_df["p50"].values
    p90 = pred_df["p90"].values

    p50 = p50.clip(p10, p90)
    p90 = np.maximum(p90, p50)
    p10 = np.minimum(p10, p50)

    pred_df["p10"] = p10
    pred_df["p50"] = p50
    pred_df["p90"] = p90
    
    pred_df = pred_df.rename(columns={
        "p10": "predicted_price(10%)",
        "p50": "predicted_price(50%)",
        "p90": "predicted_price(90%)",
    })
    print("pred_df: \n", pred_df)

    tmw_start = pd.to_datetime(today + timedelta(days=1))
    tmw_end = pd.to_datetime(today + timedelta(days=2))
    pred_tmw_only = pred_df[
        (pred_df["timestamp"] >= tmw_start) &
        (pred_df["timestamp"] < tmw_end)
    ].copy()
    pred_tmw_only["forecast_date"] = pd.to_datetime(today)
    pred_tmw_only["horizon_days"] = 1

    if PRICE_HISTORY_PATH.exists():
        hist = pd.read_parquet(PRICE_HISTORY_PATH)
    else:
        hist = pd.DataFrame()

    hist = pd.concat([hist, pred_tmw_only], axis=0, ignore_index=True)
    
    hist = hist.drop_duplicates(
        subset=["forecast_date", "timestamp", "horizon_days"],
        keep="last"
    ).sort_values(["forecast_date", "timestamp"])
    hist.to_parquet(PRICE_HISTORY_PATH, index=False)
    print(f"[SAVE] price history: {tmw_start} {PRICE_HISTORY_PATH}")
    history = pd.read_parquet(PRICE_HISTORY_PATH)

    eval_df = history.merge(
        price[["timestamp", "tokyo_price_jpy_per_kwh"]],
        on="timestamp",
        how="inner"
    ).copy()
    eval_df["forecast_date"] = pd.to_datetime(eval_df["forecast_date"])
    cutoff = pd.to_datetime(today - timedelta(days=7))

    eval_df = eval_df[eval_df["forecast_date"] >= cutoff].copy()
    print("確認: ", eval_df)
    eval_df["crps"] = eval_df.apply(row_crps, axis=1)
    eval_df["timestamp_30min"] = eval_df["timestamp"].dt.strftime("%H:%M")
    eval_df.to_parquet(PRICE_EVAL_DETAIl_PATH, index=False)
    print(f"[SAVE] price evaluation detail to {PRICE_EVAL_DETAIl_PATH}")
    
    crps_30min = (
        eval_df.groupby("timestamp_30min")["crps"]
        .mean()
        .reset_index()
    )
    
    # 需給実績をくっつけて可視化で使用する(timestamp, predicted_demand, price_realized)    
    out = pred_df.merge(
        price[["timestamp", "tokyo_price_jpy_per_kwh"]],
        on="timestamp",
        how="outer"
    )
    print("final_out:\n", out)
    print("final_eval:\n", crps_30min)
    
    out.to_parquet(PRICE_OUT_PATH, index=False)
    print(f"[SAVE] price forecast to {PRICE_OUT_PATH}")
    crps_30min.to_parquet(PRICE_EVAL_OUT_PATH, index=False)
    print(f"[SAVE] price evaluation to {PRICE_EVAL_OUT_PATH}")

if __name__ == "__main__":
    predict_price()
