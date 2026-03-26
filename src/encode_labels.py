import json

LABELS_PATH = "data/relationship_dataset/labels.json"
OUTPUT_PATH = "data/relationship_dataset/labels_encoded.json"
MAP_PATH = "data/relationship_dataset/label_map.json"

with open(LABELS_PATH) as f:
    labels = json.load(f)

predicates = sorted(list(set([item["predicate"] for item in labels])))

label_map = {p: i for i, p in enumerate(predicates)}

encoded = []

for item in labels:
    encoded.append({
        "image": item["image"],
        "label": label_map[item["predicate"]]
    })

with open(OUTPUT_PATH, "w") as f:
    json.dump(encoded, f)

with open(MAP_PATH, "w") as f:
    json.dump(label_map, f)

print("Classes:", len(label_map))