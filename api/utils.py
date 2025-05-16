import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json
import os
import urllib.request

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

# Загрузка словаря классов ImageNet
def load_imagenet_classes():
    # Проверяем, есть ли уже кешированный файл
    cache_file = "imagenet_classes.txt"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    else:
        # Загружаем с официального репозитория
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        
        with urllib.request.urlopen(url) as f:
            classes = [line.decode('utf-8').strip() for line in f.readlines()]
            
        # Кешируем для будущего использования
        with open(cache_file, 'w') as f:
            for class_name in classes:
                f.write(f"{class_name}\n")
                
        return classes

# Форматирование предсказаний для вывода
def format_predictions(predictions, top_k=5):
    # Загружаем названия классов
    imagenet_classes = load_imagenet_classes()
    
    # Получаем топ K предсказаний
    probabilities, indices = torch.topk(predictions, top_k)
    
    results = []
    for i, (prob, idx) in enumerate(zip(probabilities.tolist()[0], indices.tolist()[0])):
        class_idx = int(idx)  # Преобразуем в обычное число
        results.append({
            "rank": i + 1,
            "class_id": str(class_idx),
            "class_name": imagenet_classes[class_idx],  # Используем словесное название
            "probability": float(prob)
        })
    
    return results
