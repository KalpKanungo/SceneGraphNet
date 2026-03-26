import json
import os
import cv2
from tqdm import tqdm

SUBSET_PATH = "data/relationship_dataset/subset.json"
IMAGE_MAP_PATH = "data/relationship_dataset/image_paths.json"

OUTPUT_DIR = "data/relationship_dataset/images"
LABELS_PATH = "data/relationship_dataset/labels.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(SUBSET_PATH) as f:
    subset = json.load(f)

with open(IMAGE_MAP_PATH) as f:
    image_map = json.load(f)

labels = []

idx = 0

for item in tqdm(subset):
    image_id = str(item["image_id"])

    if image_id not in image_map:
        continue

    img_path = image_map[image_id]
    img = cv2.imread(img_path)

    if img is None:
        continue

    h_img, w_img, _ = img.shape

    s = item["subject"]
    o = item["object"]

    x1 = min(s["x"], o["x"])
    y1 = min(s["y"], o["y"])
    x2 = max(s["x"] + s["w"], o["x"] + o["w"])
    y2 = max(s["y"] + s["h"], o["y"] + o["h"])

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    crop = cv2.resize(crop, (128, 128))

    filename = f"{idx}.jpg"
    save_path = os.path.join(OUTPUT_DIR, filename)

    cv2.imwrite(save_path, crop)

    labels.append({
        "image": filename,
        "predicate": item["predicate"]
    })

    idx += 1

with open(LABELS_PATH, "w") as f:
    json.dump(labels, f)

print("Total processed:", idx)