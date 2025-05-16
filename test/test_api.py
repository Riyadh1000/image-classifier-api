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
    import numpy as np
    import torch
    
    # Создаем мок-объект для модели
    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.2, 0.3, 0.5]])
    
    # Патчим функцию load_model
    monkeypatch.setattr("api.main.load_model", lambda: mock_model)
    
    # Создаем тестовое изображение
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    files = {"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/classify/", files=files)
    
    assert response.status_code == 200
    assert "status" in response.json()
    assert "predictions" in response.json()
    assert response.json()["status"] == "success"
    
    predictions = response.json()["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) <= 5
