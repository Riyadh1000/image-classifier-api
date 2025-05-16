import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image
import torch
import numpy as np

# Импорт приложения
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from api.main import app


client = TestClient(app)

@pytest.fixture
def mock_image():
    # Создаем тестовое изображение в памяти
    image = Image.new('RGB', (100, 100), color='red')
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    return image_bytes

def test_predict(mock_image):
    # Мокируем torch операции вместо использования реальной модели
    with patch('torch.nn.functional.softmax', return_value=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])), \
         patch('torch.topk', return_value=(torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1]), 
                                          torch.tensor([4, 3, 2, 1, 0]))):
        
        response = client.post(
            "/predict",
            files={"file": ("test_image.jpg", mock_image, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 5
        
        # Проверяем структуру ответа
        for pred in data["predictions"]:
            assert "class" in pred
            assert "probability" in pred
            # Проверяем, что вероятность в правильном диапазоне
            assert 0 <= pred["probability"] <= 1

def test_predict_invalid_file():
    # Тестируем с неверным форматом файла
    bad_file = io.BytesIO(b"not an image")
    response = client.post(
        "/predict",
        files={"file": ("bad_file.txt", bad_file, "text/plain")}
    )
    assert response.status_code == 422  # Unprocessable Entity
