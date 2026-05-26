import pandas as pd
from scipy.stats import ks_2samp
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "campaign"

def generate_drift_report():
    if not (DATA_DIR / "campaign_data.csv").exists():
        print("Data not found. Run generate_sample_data.py first.")
        return
        
    df = pd.read_csv(DATA_DIR / "campaign_data.csv")
    
    # Simulate production data with slight drift
    prod_df = df.copy()
    prod_df["income"] = prod_df["income"] * 1.15 # 15% drift in income
    prod_df["age"] = prod_df["age"] + 3
    
    drift_report = {"summary": "Data Drift Analysis (Kolmogorov-Smirnov Test)", "features": {}}
    
    for col in ["age", "income", "num_campaigns", "recency_days", "tenure_days"]:
        stat, p_value = ks_2samp(df[col], prod_df[col])
        drift_report["features"][col] = {
            "p_value": round(p_value, 5),
            "drift_detected": bool(p_value < 0.05)
        }
        
    report_path = BASE_DIR / "monitoring" / "drift_report.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(drift_report, f, indent=2)
        
    print(f"[OK] ML Data Drift report generated at {report_path}")

if __name__ == "__main__":
    generate_drift_report()
