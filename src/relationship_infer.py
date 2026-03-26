import torch
import cv2
import json
import os
from src.model import RelationshipNet
from huggingface_hub import hf_hub_download

# ========================
# CONFIG
# ========================
MODEL_REPO = "kalpkanungo/scenegraphnet-relationship-model"
MODEL_FILENAME = "relationship_model.pth"

LABEL_MAP_PATH = "data/relationship_dataset/label_map.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# LOAD LABEL MAP
# ========================
if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
else:
    print("⚠️ label_map.json not found, using fallback")
    label_map = {
        "0": "on",
        "1": "next to",
        "2": "under"
    }

inv_map = {int(k): v for k, v in label_map.items()}
num_classes = len(label_map)

# ========================
# LOAD MODEL FROM HF
# ========================
model = RelationshipNet(num_classes)

try:
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME
    )
    print("✅ Model downloaded from Hugging Face")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

except Exception as e:
    print(f"⚠️ Failed to load model from HF: {e}")
    model = None

# ========================
# PREDICTION FUNCTION
# ========================
def predict(image):
    # Fallback if model not loaded
    if model is None:
        return "next to"

    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = (image - 0.5) / 0.5

    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    return inv_map.get(pred, "unknown")