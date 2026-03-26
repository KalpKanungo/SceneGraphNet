import json
import os

INPUT_PATH = "data/relationship_dataset/subset.json"
OUTPUT_PATH = "data/relationship_dataset/image_paths.json"

IMAGE_DIR_1 = "data/visual_genome/images/VG_100K"
IMAGE_DIR_2 = "data/visual_genome/images2/VG_100K_2"

with open(INPUT_PATH) as f:
    data = json.load(f)

image_ids = set([item["image_id"] for item in data])

image_map = {}

for img_id in image_ids:
    filename = f"{img_id}.jpg"

    path1 = os.path.join(IMAGE_DIR_1, filename)
    path2 = os.path.join(IMAGE_DIR_2, filename)

    if os.path.exists(path1):
        image_map[img_id] = path1
    elif os.path.exists(path2):
        image_map[img_id] = path2

with open(OUTPUT_PATH, "w") as f:
    json.dump(image_map, f)

print("Total images found:", len(image_map))