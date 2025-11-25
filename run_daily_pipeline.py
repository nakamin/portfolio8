"""
毎日 GitHub Actions から呼び出される統合パイプライン
"""

from pathlib import Path
import json
from datetime import datetime, timezone, timedelta

from fetch_weather import fetch_weather
from fetch_market import fetch_market
from fetch_price import fetch_price
from fetch_demand import fetch_demand
from predict_demand import predict_demand
from predict_price import predict_price
from optimize_dispatch import optimize_dispatch

CACHE_DIR = Path("data/cache")

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ========== 1. データ取得 ==========
    print("\n===== weather =====\n")
    fetch_weather()
    print("\n===== market =====\n")
    fetch_market()
    print("\n===== price =====\n")
    fetch_price()
    print("\n===== demand =====\n")
    fetch_demand()

    # ========== 2. 需要予測 ==========
    print("\n===== predict demand =====\n")
    predict_demand()

    # ========== 3. 価格予測 ==========
    print("\n===== predict price =====\n")
    predict_price()

    # ========== 4. 電源構成最適化 ==========
    print("\n===== optimize =====\n")
    optimize_dispatch()

    # ========== 5. メタ情報（更新時刻＋出典） ==========
    meta = {
        "last_updated_jst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "sources": {
            "weather": "Open-Meteo JMA API",
            "markets": {
                "fx": "Frankfurter (ECB)",
                "oil": "EIA / yfinance BZ=F",
                "coal": "World Bank Pink Sheet",
                "lng": "EIA / yfinance NG=F",
            },
            "jepx": "JEPX 日前スポット 東京エリア",
            "demand_model": "自作需要予測モデル",
            "price_model": "自作価格予測モデル",
            "optimizer": "自作数理最適化モデル",
        },
    }
    meta_path = CACHE_DIR / "metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Daily pipeline finished.")
    print(f"- meta:     {meta_path}")

if __name__ == "__main__":
    main()
