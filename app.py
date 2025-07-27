from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to ["http://localhost:5173"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)  # 10 classes

# Load your trained weights
model.load_state_dict(torch.load("smartbin_resnet18.pth", map_location=device))
model.eval()
model.to(device)

# Define transforms (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Define your class names (must match your dataset.classes order!)
class_names = [
    "batteries",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        input_tensor = transform(img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_idx = torch.max(probs, dim=0)

        predicted_label = class_names[top_idx]

        return JSONResponse({
            "label": predicted_label,
            "confidence": round(top_prob.item(), 4)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})
