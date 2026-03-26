import torch
import cv2
import json
from src.model import RelationshipNet

MODEL_PATH = "models/relationship_model.pth"
LABEL_MAP_PATH = "data/relationship_dataset/label_map.json"

device = "mps" if torch.backends.mps.is_available() else "cpu"

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

inv_map = {v: k for k, v in label_map.items()}

num_classes = len(label_map)

model = RelationshipNet(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def predict(image):
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = (image - 0.5) / 0.5

    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    return inv_map[pred]