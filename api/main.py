from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../model/tomatoes.h5")

CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "Tomato_healthy"
]

@app.get("/")
async def root():
    return {"message": "Tomato Disease API Running"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(io.BytesIO(data))
    image = image.resize((256, 256))  # match your training size
    image = np.array(image)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = read_file_as_image(await file.read())

    # convert type
    image = image.astype(np.float32)

    

    # add batch dimension
    image = np.expand_dims(image, axis=0)

    # predict
    predictions = MODEL.predict(image)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
