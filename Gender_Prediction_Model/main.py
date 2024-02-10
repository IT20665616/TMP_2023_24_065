from tensorflow import keras
import numpy as np
from fastapi import FastAPI
from keras.preprocessing import image
from keras.models import load_model

app = FastAPI()



# Load the saved model
loaded_model = load_model("gender_model.h5")

@app.post("/predict")
async def gender_predict(imagePath: str):
    img = image.load_img(imagePath, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = loaded_model.predict(img_array)

    # Convert the prediction to a binary label (0 or 1) based on a threshold (e.g., 0.5)
    binary_prediction = 1 if predictions[0][0] > 0.5 else 0

    # Return the prediction
    return 'Female' if binary_prediction == 1 else 'Male'