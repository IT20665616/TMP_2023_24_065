import shutil
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from tensorflow import keras
from keras.models import load_model
from fastapi import UploadFile, File

app = FastAPI()

age_model = load_model("age_predict.h5")
age_class_labels = ['0-20', '20-40', '40-60', '60-80']

def predict_image_model(model, image_path, class_names, target_size):
    loaded_image = keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = keras.preprocessing.image.img_to_array(loaded_image)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    output_class = class_names[np.argmax(predictions)]
    return output_class

@app.post("/age-predict")
async def age_predict(file: UploadFile = File(...)):
    with open("tempfile.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the file with your model
    return predict_image_model(age_model, "tempfile.jpg", age_class_labels, (180, 180))
