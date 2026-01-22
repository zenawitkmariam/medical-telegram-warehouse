from sqlalchemy import create_engine
import pandas as pd

DB_URI = "postgresql://postgres:root@localhost:5432/medical_db"
engine = create_engine(DB_URI)

df = pd.read_csv("data/processed/yolo_categorized.csv")

df.to_sql("yolo_image_detections", engine, schema="raw", if_exists="replace", index=False)
print("YOLO results loaded into raw.yolo_image_detections")
