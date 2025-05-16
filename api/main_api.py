from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io
import torch
import torchvision.transforms as transforms

app = FastAPI()

# Загрузка модели (можно заменить на свою)
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
model.eval()

# Классы ImageNet (можно заменить на свои)
IMAGENET_CLASSES = [f"class_{i}" for i in range(1000)]

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Invalid image file")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process the file")

    input_tensor = transform_image(image_bytes)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

    predictions = [
        {
            "class": IMAGENET_CLASSES[catid],
            "probability": prob.item()
        }
        for prob, catid in zip(top5_prob, top5_catid)
    ]
    return JSONResponse(content={"predictions": predictions})
