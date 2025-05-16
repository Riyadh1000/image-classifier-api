from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
import io
from .utils import preprocess_image, format_predictions

app = FastAPI(title="Классификатор изображений API (ShuffleNet)")

model = None

@app.on_event("startup")
async def startup():
    load_model()

def load_model():
    global model
    if model is None:
        try:
            # Загрузка ShuffleNetV2 вместо MobileNetV2
            model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
            model.eval()  # установить режим оценки
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            raise e
    return model

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # изменить в продакшене
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify/", response_class=JSONResponse)
async def classify_image(file: UploadFile = File(...)):
    """
    Загрузите изображение для классификации.
    Возвращает список из 5 наиболее вероятных классов.
    """
    allowed_extensions = ["jpg", "jpeg", "png"]
    file_extension = file.filename.split(".")[-1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Файл должен быть одного из следующих типов: {', '.join(allowed_extensions)}"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        processed_image = preprocess_image(image)

        model = load_model()

        with torch.no_grad():
            logits = model(processed_image)
            probabilities = F.softmax(logits, dim=1)

        results = format_predictions(probabilities)

        return {
            "status": "success",
            "predictions": results
        }

    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Невалидный формат изображения")
    except Exception as e:
        print(f"Ошибка при обработке изображения: {str(e)}")  # Для отладки
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")


@app.get("/", response_class=JSONResponse)
async def root():
    """
    Корневой эндпоинт с информацией о API.
    """
    return {
        "name": "Классификатор изображений API",
        "description": "API для классификации изображений с помощью ShuffleNetV2",
        "endpoints": {
            "/classify/": "Загрузка и классификация изображений"
        }
    }
