from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import torch
import json

# Load the model from the saved path
def model_fn(model_dir):
    model = YOLO(f"{model_dir}/yolov11_best.pt")
    return model

# Convert the incoming request to a PIL image
def input_fn(request_body, content_type='application/octet-stream'):
    if content_type == 'application/octet-stream':
        return Image.open(BytesIO(request_body)).convert("RGB")
    raise Exception(f"Unsupported content type: {content_type}")

# Run inference and return the detections
def predict_fn(input_data, model):
    results = model(input_data, conf=0.5)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        conf = float(box.conf[0])
        detections.append({
            "name": name,
            "confidence": round(conf, 3),
            "bbox": [round(coord, 2) for coord in box.xyxy[0].tolist()]
        })
    return detections

# Format the output as JSON
def output_fn(prediction, accept='application/json'):
    return json.dumps(prediction)
