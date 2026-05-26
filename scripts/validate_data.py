import pandas as pd
import pandera as pa
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CAMPAIGN_FILE = BASE_DIR / "data" / "campaign" / "campaign_data.csv"

def validate_campaign_data():
    if not CAMPAIGN_FILE.exists():
        print(f"Error: {CAMPAIGN_FILE} does not exist. Run scripts/generate_sample_data.py first.")
        sys.exit(1)
        
    df = pd.read_csv(CAMPAIGN_FILE)
    
    # R3: 5+ business rules, missing values, schema checks
    schema = pa.DataFrameSchema({
        "age": pa.Column(int, pa.Check.in_range(18, 100), nullable=False),
        "income": pa.Column(int, pa.Check.ge(0), nullable=False),
        "num_campaigns": pa.Column(int, pa.Check.ge(1), nullable=False),
        "num_clicks": pa.Column(int, pa.Check.ge(0), nullable=False),
        "num_opens": pa.Column(int, pa.Check.ge(0), nullable=False),
        "recency_days": pa.Column(int, pa.Check.ge(1), nullable=False),
        "tenure_days": pa.Column(int, pa.Check.ge(0), nullable=False),
        "gender": pa.Column(str, pa.Check.isin(["M", "F", "Other"]), nullable=False),
        "channel": pa.Column(str, nullable=False),
        "product_category": pa.Column(str, nullable=False),
        "region": pa.Column(str, nullable=False),
        "converted": pa.Column(int, pa.Check.isin([0, 1]), nullable=False)
    })
    
    try:
        schema.validate(df)
        print("[OK] Data validation passed. No missing values, correct schema, and business rules passed.")
    except pa.errors.SchemaError as exc:
        print(f"[FAIL] Data validation failed:\n{exc}")
        sys.exit(1)

if __name__ == "__main__":
    validate_campaign_data()
