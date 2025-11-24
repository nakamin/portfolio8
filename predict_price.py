from pathlib import Path
import joblib
import pandas as pd
from datetime import datetime, timedelta, timezone

import lightgbm as lgb
from typing import Optional, Any
from huggingface_hub import hf_hub_download

from utils.make_date_features import make_date, make_daypart, make_season

CACHE_DIR = Path("data/cache")
HF_REPO_ID = "nakamichan/power-forecast-models"

# 実績と予報
DEMAND_PATH = CACHE_DIR / "demand_forecast.parquet" # 実績も入っている
WEATHER_PATH = CACHE_DIR / "weather_bf1w_af1w.parquet"
MARKET_PATH = CACHE_DIR / "fx_commodity_30min_af1w.parquet"
PRICE_PATH = CACHE_DIR / "spot_tokyo_bf1w_tdy.parquet"

# モデル
GAM_PATH = "gam_price.pkl"
LGB10_PATH = "lgb_price_p10.txt"
LGB50_PATH = "lgb_price_p50.txt"
LGB90_PATH = "lgb_price_p90.txt"

# このモデルの結果
PRICE_OUT_PATH = CACHE_DIR / "price_forecast.parquet"

GAM_FEATURES = ["day_of_year", "season", "is_holiday",
                "USDJPY", "brent", "henry_hub", "coal_aus",
                "temp_7d_ma", "demand_7d_ma", "cooling_degree", "heating_degree"]

LGB_FEATURES = [
    "demand",
    "temperature", "humidity", "wind_speed", "shortwave_radiation","sunshine_duration",
    "price_lag_24h", "price_lag_48h", "price_lag_72h", "price_lag_1w",
    "hour", "daypart",
    "y_gam",
    "day_of_year", "season", "is_holiday",
    "USDJPY", "brent", "henry_hub", "coal_aus",
    "temp_7d_ma", "demand_7d_ma",
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
    gam = joblib.load(gam_path)

    lgb_p10 = lgb.Booster(model_file=lgb10_path)
    lgb_p50 = lgb.Booster(model_file=lgb50_path)
    lgb_p90 = lgb.Booster(model_file=lgb90_path)
    
    return gam, lgb_p10, lgb_p50, lgb_p90

def build_price_features(
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    market: pd.DataFrame,
    price: pd.DataFrame,
    tomorrow
) -> pd.DataFrame:
    """

    """
    # timestamp を必ず datetime にしておく
    demand["timestamp"] = pd.to_datetime(demand["timestamp"])
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    market["timestamp"] = pd.to_datetime(market["timestamp"])
    price["timestamp"] = pd.to_datetime(price["timestamp"])

    # 結合
    merged_df = demand.merge(weather, on="timestamp", how="left") \
                    .merge(market, on="timestamp", how="left") \
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
    
    merged_df["day_of_year"] = merged_df["timestamp"].dt.dayofyear
    # 過去7日移動平均の需要
    merged_df["demand_7d_ma"] = merged_df["demand"].rolling(window=48*7, min_periods=1).mean()
    # 過去7日移動平均の気温
    merged_df["temp_7d_ma"] = merged_df["temperature"].rolling(window=48*7, min_periods=1).mean()
    # 冷房・暖房負荷の指標
    merged_df["cooling_degree"] = (merged_df["temperature"] - 22).clip(lower=0)
    merged_df["heating_degree"] = (18 - merged_df["temperature"]).clip(lower=0)
    # 価格ラグ
    merged_df["price_lag_24h"] = merged_df["tokyo_price_jpy_per_kwh"].shift(48)
    merged_df["price_lag_48h"] = merged_df["tokyo_price_jpy_per_kwh"].shift(96)
    merged_df["price_lag_72h"] = merged_df["tokyo_price_jpy_per_kwh"].shift(144)
    merged_df["price_lag_1w"] = merged_df["tokyo_price_jpy_per_kwh"].shift(336)
    
    
    pred_time_df = merged_df[merged_df["timestamp"] >= tomorrow]
    pred_time = pred_time_df["timestamp"]
    print("prediction time: \n", pred_time)
    
    return merged_df, pred_time


def predict_gam(
    gam_model: Optional[Any],
    test_df
) -> pd.DataFrame:
    """
    GAMモデルで推論し、y_gamをtest_dfに代入してLightGBMでも使用
    """
    
    X_gam_test = test_df.copy()
    X_gam_test["season"] = X_gam_test["season"].cat.codes.astype("int8")
    X_gam_test = X_gam_test[GAM_FEATURES].to_numpy()
    print("X_gam_test: \n", X_gam_test)
    test_df["y_gam"] = gam_model.predict(X_gam_test)

    return test_df

def predict_lgb(
    lgb_p10, lgb_p50, lgb_p90,test_df
) -> pd.DataFrame:
    """

    """
    
    print(test_df.columns)
    X_lgb_test = test_df[LGB_FEATURES]

    r10_test = lgb_p10.predict(X_lgb_test)
    r50_test = lgb_p50.predict(X_lgb_test)
    r90_test = lgb_p90.predict(X_lgb_test)

    return r10_test, r50_test, r90_test


def predict_price():
    
    # 特徴量の読み込み
    demand = pd.read_parquet(DEMAND_PATH)
    print("demand\n", demand)
    weather = pd.read_parquet(WEATHER_PATH)
    print("weather\n", weather)
    market = pd.read_parquet(MARKET_PATH)
    print("market\n", market)
    price = pd.read_parquet(PRICE_PATH)
    print("price\n", price)
    
    JST = timezone(timedelta(hours=9))
    today = pd.Timestamp(datetime.now(JST).date())
    tomorrow = today + timedelta(days=1)

    features_df, pred_time = build_price_features(demand, weather, market, price, tomorrow)

    test_df = features_df[features_df["timestamp"] >= tomorrow].reset_index(drop=True)
    
    gam, lgb_p10, lgb_p50, lgb_p90 = load_price_model()
    
    test_df = predict_gam(gam, test_df)
    
    r10_test, r50_test, r90_test = predict_lgb(lgb_p10, lgb_p50, lgb_p90, test_df)

    # 予測結果をparquetで保存
    pred_df = pd.DataFrame({
        "timestamp": pred_time,
        "predicted_price(10%)": r10_test.flatten(),
        "predicted_price(50%)": r50_test.flatten(),
        "predicted_price(90%)": r90_test.flatten(),
    }).reset_index(drop=True)
    
    print("pred_df: \n", pred_df)
    
    # 需給実績をくっつけて可視化で使用する(timestamp, predicted_demand, price_realized)    
    out = pred_df.merge(
        price[["timestamp", "tokyo_price_jpy_per_kwh"]],
        on="timestamp",
        how="outer",
    )
    out["price"] = out["predicted_price(50%)"].fillna(price["tokyo_price_jpy_per_kwh"])
    print("final_out:\n", out)
    
    out.to_parquet(PRICE_OUT_PATH, index=False)
    print(f"[SAVE] price forecast to {PRICE_OUT_PATH}")

if __name__ == "__main__":
    predict_price()
