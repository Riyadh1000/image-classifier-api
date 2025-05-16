import sys
import os
from unittest.mock import patch, MagicMock
import torch

# Добавляем корневой каталог проекта в пути Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Мокируем модель перед импортом приложения
@pytest.fixture(autouse=True, scope="session")
def mock_model():
    with patch('torchvision.models.resnet50') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock_instance.return_value = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        yield mock
