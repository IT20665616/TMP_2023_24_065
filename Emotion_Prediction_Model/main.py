import os
import shutil
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()

# Load the trained emotion recognition model
voice_model = load_model('voice_model.h5')


def predict_audio_model(model, audio_path, labels, segment_length, overlap, sampling_rate):
    y, _ = librosa.load(audio_path, sr=sampling_rate)
    emotion_predictions = []

    for start in np.arange(0, len(y), int((segment_length - overlap) * sampling_rate)):
        end = int(start + segment_length * sampling_rate)
        segment = y[start:end]
        mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sampling_rate, n_mfcc=40).T, axis=0)
        mfcc = mfcc.reshape(1, mfcc.shape[0], 1)
        emotion_probabilities = model.predict(mfcc)
        emotion_predictions.append(emotion_probabilities)

    best_matching_emotion_index = np.argmax(np.sum(emotion_predictions, axis=0))
    best_matching_emotion_label = labels[best_matching_emotion_index]
    return best_matching_emotion_label


@app.post("/emotion-prediction")
async def predict_emotion_from_audio(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary file
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Now process this temporary file with your model
    emotion_label = predict_audio_model(voice_model, temp_file_path, ['Angry', 'Neutral', 'Sad', 'Happy'], 3, 0.0, 22050)

    # Clean up the temporary file
    os.remove(temp_file_path)

    # Return the prediction result
    return emotion_label