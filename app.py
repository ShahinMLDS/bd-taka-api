from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

model = YOLO("model/best.pt")

@app.get("/")
def home():
    return {"message": "Taka detection API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = model.predict(image)

    detections = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()

        detections.append({
            "class": model.names[cls],
            "confidence": conf,
            "bbox": bbox
        })

    return {"detections": detections}