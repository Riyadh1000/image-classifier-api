import sys
import os
import pytest  # Добавляем импорт pytest
from unittest.mock import patch, MagicMock
import torch


# Мокируем модель перед импортом приложения
@pytest.fixture(autouse=True, scope="session")
def mock_model():
    with patch('torchvision.models.resnet50') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock_instance.return_value = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        yield mock
