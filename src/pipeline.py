import cv2
from itertools import combinations

from src.detection import detect
from src.relationship_infer import predict
from src.spatial_rules import get_relation


def run_pipeline(image):

    detections = detect(image)

    results = []
    pair_seen = set()

    for obj1, obj2 in combinations(detections, 2):

        # 🔥 avoid duplicate object pairs (by label)
        pair_key = (obj1["label"], obj2["label"])
        if pair_key in pair_seen:
            continue
        pair_seen.add(pair_key)

        # boxes: [x1, y1, x2, y2]
        x1, y1, x2, y2 = obj1["box"]
        x3, y3, x4, y4 = obj2["box"]

        # convert to (x, y, w, h)
        box1 = (x1, y1, x2 - x1, y2 - y1)
        box2 = (x3, y3, x4 - x3, y4 - y3)

        # 🔥 spatial relation (primary)
        spatial_rel = get_relation(box1, box2)

        # crop region for model
        x_min = int(min(x1, x3))
        y_min = int(min(y1, y3))
        x_max = int(max(x2, x4))
        y_max = int(max(y2, y4))

        crop = image[y_min:y_max, x_min:x_max]

        if crop.size == 0:
            continue

        # 🔥 model prediction (secondary)
        try:
            model_rel = predict(crop)
        except:
            model_rel = None

        # 🔥 hybrid logic
        if model_rel in ["holding", "sitting_on"]:
            relation = model_rel
        else:
            relation = spatial_rel

        results.append({
            "subject": obj1["label"],
            "object": obj2["label"],
            "relation": relation
        })

    # 🔥 remove duplicate relations
    unique = set()
    clean_results = []

    for r in results:
        key = (r["subject"], r["object"], r["relation"])
        if key not in unique:
            unique.add(key)
            clean_results.append(r)

    # 🔥 limit number of relations (clean graph)
    MAX_RELATIONS = 8
    clean_results = clean_results[:MAX_RELATIONS]

    return clean_results