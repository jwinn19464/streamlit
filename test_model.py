# test_model.py
from model_utils import load_model, preprocess_audio, predict_emotion

def test_prediction(audio_file_path):
    # Load model
    model = load_model()
    if model is None:
        return "Failed to load model"
    
    # Preprocess audio
    features = preprocess_audio(audio_file_path)
    if features is None:
        return "Failed to preprocess audio"
    
    # Make prediction
    prediction = predict_emotion(model, features)
    if prediction is None:
        return "Failed to make prediction"
    
    return prediction

# Test with an audio file
result = test_prediction("03-01-01-01-01-01-01.wav")
print(result)
