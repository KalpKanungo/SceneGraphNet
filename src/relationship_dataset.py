import json
import random
from tqdm import tqdm
from src.config import RELATIONS

INPUT_PATH = "data/visual_genome/region_graphs.json"
OUTPUT_PATH = "data/relationship_dataset/subset.json"

subset_size = 10000
def normalize_predicate(p):
    if "on" in p:
        return "on"
    if "next_to" in p or "next" in p:
        return "next_to"
    if "hold" in p:
        return "holding"
    if "ride" in p:
        return "riding"
    if "behind" in p:
        return "behind"
    if "front" in p:
        return "in_front_of"
    if "under" in p:
        return "under"
    return None

with open(INPUT_PATH) as f:
    data = json.load(f)

valid_samples = []

for item in tqdm(data):
    image_id = item["image_id"]

    for region in item.get("regions", []):
        objects = region.get("objects", [])
        obj_map = {obj["object_id"]: obj for obj in objects}

        for rel in region.get("relationships", []):
            predicate = rel.get("predicate", "").lower().replace(" ", "_")

            normalized = normalize_predicate(predicate)

            if normalized is not None:
                subject_id = rel.get("subject_id")
                object_id = rel.get("object_id")

                if subject_id in obj_map and object_id in obj_map:
                    subject = obj_map[subject_id]
                    obj = obj_map[object_id]

                    valid_samples.append({
                        "image_id": image_id,
                        "predicate": normalized,
                        "subject": subject,
                        "object": obj
                    })

random.shuffle(valid_samples)

subset = valid_samples[:subset_size]

with open(OUTPUT_PATH, "w") as f:
    json.dump(subset, f)

print("Total samples:", len(subset))