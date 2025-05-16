import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json
import os

# Предобработка изображений для ShuffleNet
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    image_tensor = preprocess(image)
    return image_tensor.unsqueeze(0)  # добавляем размерность батча

# Форматирование предсказаний для вывода
def format_predictions(predictions, top_k=5):
    # Получаем топ K предсказаний
    probabilities, indices = torch.topk(predictions, top_k)
    
    # Здесь можно загрузить метки ImageNet из файла
    # Для простоты используем общие имена классов
    
    results = []
    for i, (prob, idx) in enumerate(zip(probabilities.tolist()[0], indices.tolist()[0])):
        results.append({
            "rank": i + 1,
            "class_id": str(idx.item()),
            "class_name": f"class_{idx.item()}",
            "probability": float(prob)
        })
    
    return results
