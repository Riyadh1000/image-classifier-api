import pytest
from fastapi.testclient import TestClient
import os
from api.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()
    assert "description" in response.json()
    assert "endpoints" in response.json()

def test_classify_endpoint_wrong_extension():
    # Тестирование с неверным расширением файла
    files = {"file": ("test.txt", b"content", "text/plain")}
    response = client.post("/classify/", files=files)
    assert response.status_code == 400

def test_classify_endpoint_invalid_image():
    # Тестирование с невалидным изображением
    files = {"file": ("test.jpg", b"not an image", "image/jpeg")}
    response = client.post("/classify/", files=files)
    assert response.status_code in [422, 500]  # Может вернуть разные ошибки в зависимости от обработки

# Мок-тест для проверки успешного ответа (требует изменения для работы с реальным изображением)
def test_classify_endpoint_mock_success(monkeypatch):
    from unittest.mock import MagicMock
    import io
    from PIL import Image
    import torch
    import numpy as np  # Явно импортируем numpy
    
    # Создаем более полный мок-объект для модели
    mock_model = MagicMock()
    mock_output = torch.tensor([[0.1, 0.2, 0.3, 0.25, 0.15]])
    mock_model.return_value = mock_output
    
    # Патчим не только load_model, но и format_predictions
    monkeypatch.setattr("api.main.load_model", lambda: mock_model)
    
    # Также патчим format_predictions, чтобы он возвращал правильный формат данных
    monkeypatch.setattr(
        "api.main.format_predictions",
        lambda pred, top_k=5: [
            {
                "rank": i+1,
                "class_id": str(i),
                "class_name": f"class_{i}",
                "probability": float(0.9 - i*0.1)
            }
            for i in range(5)
        ]
    )
    
    # Создаем простое тестовое изображение
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    # Отправляем запрос
    files = {"file": ("test.jpg", img_byte_arr.read(), "image/jpeg")}
    response = client.post("/classify/", files=files)
    
    assert response.status_code == 200
    assert "status" in response.json()
    assert "predictions" in response.json()