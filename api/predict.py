from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from io import BytesIO

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Load class dictionary
with open("requirements.txt", "r") as f:
    class_dict = json.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        # Preprocess image
        img = img.resize((299, 299))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        # Make prediction
        predictions = model.predict(img_array)
        probs = predictions[0]
        # Get the index of the highest probability
        max_index = int(np.argmax(probs))
        # Map the index to the corresponding label
        predicted_label = class_dict[str(max_index)]
        return JSONResponse(content={"prediction": max_index, "label": predicted_label})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
