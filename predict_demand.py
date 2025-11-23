import os
import pandas as pd
from typing import Any
from pathlib import Path
from datetime import datetime, timedelta, timezone
from torch.utils.data import DataLoader, TensorDataset
import torch
import joblib
from huggingface_hub import hf_hub_download

from model.demand_model import GRUModel
from utils.make_date_features import make_date

CACHE_DIR = Path("data/cache")

WEATHER_PATH = CACHE_DIR / "weather_bf1w_af1w.parquet"
DEMAND_REALIZED_PATH = CACHE_DIR / "demand_bf1w_ytd.parquet"
DEMAND_OUT_PATH = CACHE_DIR / "demand_forecast.parquet"

HF_REPO_ID = "nakamichan/power-forecast-models"
DEMAND_MODEL = "model_demand.pth"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"

# =========================
# モデルのロード部分
# =========================

def load_demand_model() -> Any:
    """
    需要予測モデルを読み込んで返す
    """
    # if model_path is None:
    #     model_path = os.path.join(MODEL_DIR, "model_demand.pth")
    
    # モデルファイルを Hub からダウンロード
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=DEMAND_MODEL,
    )
    
    model = GRUModel()      
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return model


# =========================
# 特徴量作成部分
# =========================

def _today_jst():
    JST = timezone(timedelta(hours=9))
    return datetime.now(JST).date()

def build_demand_features(weather: pd.DataFrame, sequence_length: int) -> pd.DataFrame:
    """
    気象データから需要予測用の特徴量を作る

    入力:
        weather: weather_1w.parquet を読んだ DataFrame
        特徴量：use_col = ["year", "month", "hour", "datetime", "is_holiday", "temperature"]

    出力:
        features: モデルに入力する特徴量 DataFrame
            - index または 'timestamp' で元の時間軸と対応すること
    """
    df = weather.copy()
    weather_col = ["timestamp", "temperature", "is_forecast"]
    weather_df = df[weather_col]
    
    today = _today_jst()
    cut = pd.Timestamp(f"{today} 00:00:00")
    start_time = cut - pd.Timedelta(minutes=30 * sequence_length) # 前日の24ステップを含める
    weather_df = weather_df[weather_df["timestamp"] >= start_time]
    
    feature_pred_df = make_date(weather_df)
    feature_pred_df.reset_index(drop=True, inplace=True)
    feature_col = ["month", "hour", "is_holiday", "temperature"]
    scaled_cols = ["temperature"]
    
    X_test = feature_pred_df[feature_col]
    
    # スケーリング対象以外の列
    non_scaled_cols = [col for col in feature_col if col not in scaled_cols]

    # 非スケーリング列をそのまま抽出
    X_test_rest = X_test[non_scaled_cols].copy()
    
    scaler_X_path= hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=SCALER_X_PATH,
    )
    scaler_y_path= hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=SCALER_Y_PATH,
    )
    
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    X_test_scaled = scaler_X.transform(X_test[scaled_cols])
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=scaled_cols, index=X_test.index)
    X_test_final = pd.concat([X_test_rest, X_test_scaled_df], axis=1)
    print("X_test_final shape:", X_test_final.shape)
    print("X_test_final: \n", X_test_final)
    
    pred_time_df = feature_pred_df[feature_pred_df["timestamp"] >= cut]
    pred_time = pred_time_df["timestamp"]
    print("prediction time: \n", pred_time)
    # テンソルに変換
    data_test = torch.tensor(X_test_final.values, dtype=torch.float32)
    
    return data_test, scaler_y, pred_time
    
# モデルに適した形状に変換する
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length] # モデルの入力
        sequences.append(seq)
    return torch.stack(sequences)


def evaluate(model, dataloader, scaler_y, device):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch_X in dataloader:
            batch_X = batch_X[0].to(device)
            outputs = model(batch_X) # モデルの予測値を計算
            all_predictions.append(outputs)

    predictions = scaler_y.inverse_transform(torch.cat(all_predictions).cpu().numpy()) # すべてのバッチの予測値を1つのテンソルにしてからスケールを元に戻sす
    return predictions

# Early Stopping クラス
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float('inf')
        self.trigger_times = 0

    def check(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.trigger_times = 0
            return False  # Early Stoppingしない
        else:
            self.trigger_times += 1
            return self.trigger_times >= self.patience  # 改善が見られなければTrue
        
def predict_demand():
    
    weather = pd.read_parquet(WEATHER_PATH)
    print("weather\n", weather)
    
    # timestamp を必ず datetime にしておく
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])

    # 過去の一定期間（24時間分）のデータを1つの入力シーケンとしてまとめる
    sequence_length = 24

    features_tensor, scaler_y, pred_time = build_demand_features(weather, sequence_length)
    print("features_tensor\n", features_tensor)
    print("features_tensor.shape\n", features_tensor.shape)

    # シーケンスデータ作成
    test_sequences = create_sequences(features_tensor, sequence_length)

    print(test_sequences.shape)  # (n_samples, sequence_length, num_features)

    model = load_demand_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device)
    test_loader = DataLoader(TensorDataset(test_sequences), batch_size=128, shuffle=False)
    
    predictions = evaluate(model, test_loader, scaler_y, device)
    print("predictions: \n", predictions[:5])
    
    # 予測結果をparquetで保存
    pred_df = pd.DataFrame({
        "timestamp": pred_time,
        "predicted_demand": predictions.flatten()
    }).reset_index(drop=True)
    
    print("pred_df: \n", pred_df)
    
    demand = pd.read_parquet(DEMAND_REALIZED_PATH)
    print("demand\n", demand["timestamp"])
    
    # 需給実績をくっつけて可視化で使用する(timestamp, predicted_demand, price_realized)    
    out = pred_df.merge(
        demand,
        on="timestamp",
        how="outer",
    )
    out["demand"] = out["predicted_demand"].fillna(demand["realized_demand"])
    print("final_out:\n", out)
    
    out.to_parquet(DEMAND_OUT_PATH, index=False)
    print(f"[SAVE] demand forecast to {DEMAND_OUT_PATH}")

if __name__ == "__main__":
    predict_demand()
