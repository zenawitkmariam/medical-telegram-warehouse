import os
import pandas as pd
from ultralytics import YOLO

# -----------------------------
# Step 1: Set paths
# -----------------------------
IMAGE_DIRS = [
    "data/raw/images/tenamereja",
    "data/raw/images/tikvahpharma"
]

OUTPUT_CSV = "data/processed/yolo_results.csv"
OUTPUT_CATEGORIZED_CSV = "data/processed/yolo_categorized.csv"

# -----------------------------
# Step 2: Run YOLO detection
# -----------------------------
model = YOLO("yolov8n.pt")

results_list = []

for img_dir in IMAGE_DIRS:
    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(img_dir, img_file)
            results = model(img_path)[0]

            for obj in results.boxes.data.tolist():  # xyxy, conf, cls
                x1, y1, x2, y2, conf, cls = obj
                cls_name = model.names[int(cls)]
                results_list.append({
                    "image_file": img_file,
                    "detected_object": cls_name,
                    "confidence_score": conf
                })

# Save raw YOLO results
df = pd.DataFrame(results_list)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"YOLO detection results saved to {OUTPUT_CSV}")

# -----------------------------
# Step 3: Categorize images
# -----------------------------
def categorize(objects):
    objs = set(objects)
    if "person" in objs and ("bottle" in objs or "cell phone" in objs):
        return "promotional"
    elif "person" in objs:
        return "lifestyle"
    elif "bottle" in objs:
        return "product_display"
    else:
        return "other"

# Group by image_file
df_category = df.groupby("image_file").agg({
    "detected_object": lambda x: list(x),
    "confidence_score": "max"
}).reset_index()

df_category["image_category"] = df_category["detected_object"].apply(categorize)

# Save categorized results
df_category.to_csv(OUTPUT_CATEGORIZED_CSV, index=False)
print(f"Image categories added and saved to {OUTPUT_CATEGORIZED_CSV}")
