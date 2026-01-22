import os
import json
import pandas as pd
from sqlalchemy import create_engine, text

DB_URI = "postgresql://postgres:root@localhost:5432/medical_db"
engine = create_engine(DB_URI)

EXPECTED_COLUMNS = [
    "message_id",
    "channel_name",
    "message_date",
    "message_text",
    "has_media",
    "image_path",
    "views",
    "forwards"
]

def load_raw_data(base_path):
    all_messages = []

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_messages.extend(data)
                        else:
                            all_messages.append(data)
                except json.JSONDecodeError:
                    continue

    if not all_messages:
        print("No data found.")
        return

    df = pd.DataFrame(all_messages)
    df = df[EXPECTED_COLUMNS]

    df["message_date"] = pd.to_datetime(df["message_date"], errors="coerce")
    df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0).astype(int)
    df["forwards"] = pd.to_numeric(df["forwards"], errors="coerce").fillna(0).astype(int)
    df["ingested_at"] = pd.Timestamp.utcnow()

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw;"))

    df.to_sql(
        "telegram_messages",
        engine,
        schema="raw",
        if_exists="append",
        index=False
    )

    print(f"Loaded {len(df)} rows into raw.telegram_messages")

if __name__ == "__main__":
    load_raw_data("data/raw/telegram_messages")
