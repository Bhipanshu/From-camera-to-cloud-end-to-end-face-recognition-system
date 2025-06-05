import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import json
import os

# Load the model and modify final layer
def model_fn(model_dir):
    model = models.densenet121(pretrained=False)
    num_classes = len(os.listdir('/opt/ml/input/data/train/train'))  # class count from folder structure
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=torch.device("cpu")))
    model.eval()
    return model

# Decode image from HTTP request
def input_fn(request_body, content_type='application/octet-stream'):
    if content_type == 'application/octet-stream':
        image = Image.open(BytesIO(request_body)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        return transform(image).unsqueeze(0)  # shape: [1, 3, 112, 112]
    raise Exception(f"Unsupported content type: {content_type}")

# Run forward pass
def predict_fn(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probs[0, predicted.item()].item()
        return {
            "predicted_class": int(predicted.item()),
            "confidence": round(confidence, 3)
        }

# Return JSON result
def output_fn(prediction, accept='application/json'):
    return json.dumps(prediction)
