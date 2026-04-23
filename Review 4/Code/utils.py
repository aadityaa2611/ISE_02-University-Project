import torch
import json
from PIL import Image
import torchvision.transforms as transforms

def load_class_names():
    with open("class_names.json") as f:
        return json.load(f)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

def predict(image, model, class_names, device):
    image_tensor = preprocess_image(image).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    top3_probs, top3_indices = torch.topk(probs, 3)

    results = []
    for prob, idx in zip(top3_probs, top3_indices):
        results.append((class_names[idx.item()], float(prob)))

    return results