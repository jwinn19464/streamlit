import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import model_from_json

def load_model():
    """Load the trained model from json and weights files"""
    try:
        with open("sentiment_model.json", "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights("Data_noiseNshift.h5")
        model.compile(loss='categorical_crossentropy', 
                     optimizer='adam', 
                     metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def preprocess_audio(file_path, input_duration=4):
    """Preprocess audio file using the same steps as training"""
    try:
        # Load audio file with same parameters as training
        X, sample_rate = librosa.load(file_path, 
                                    res_type='kaiser_fast',
                                    duration=input_duration,
                                    sr=22050*2,
                                    offset=0.5)
        
        # Extract MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                            sr=sample_rate, 
                                            n_mfcc=13), 
                       axis=0)
        
        # Reshape for model
        features = mfccs.reshape(1, -1)
        return features
    except Exception as e:
        print(f"Error preprocessing audio: {str(e)}")
        return None

def predict_emotion(model, features):
    """Predict emotion from preprocessed features"""
    try:
        prediction = model.predict(features, verbose=0)
        return prediction
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None
