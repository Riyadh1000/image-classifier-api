import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from fastapi.testclient import TestClient
from api. main import app

client = TestClient(app)

def test_predict():
    with open("test/test_image.jpg", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test_image.jpg", f, "image/jpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 5
    for pred in data["predictions"]:
        assert 0 <= pred["probability"] <= 1
