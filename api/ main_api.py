from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import models, transforms
from PIL import Image
import io


#
app = FastAPI()
model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top5_probs, top5_classes = torch.topk(probabilities, 5)
    
    return {
        "predictions": [
            {"class": top5_classes[i].item(), 
             "probability": top5_probs[i].item()} 
            for i in range(top5_probs.size(0))
        ]
    }
