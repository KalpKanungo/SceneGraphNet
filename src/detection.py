from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import cv2

device = "mps" if torch.backends.mps.is_available() else "cpu"

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.to(device)
model.eval()

id2label = model.config.id2label


def detect(image):
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.shape[:2]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    from src.config import MAX_OBJECTS, CONF_THRESHOLD

    detections = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score.item() > CONF_THRESHOLD:
            detections.append({
                "label": id2label[label.item()],
                "score": score.item(),
                "box": box.tolist()
            })

    detections = sorted(detections, key=lambda x: x["score"], reverse=True)[:MAX_OBJECTS]

    return detections