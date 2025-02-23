import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import model_from_json
from audio_recorder_streamlit import audio_recorder
import time

@st.cache_resource  # This decorator helps cache the model to avoid reloading
def load_model():
    """Load the trained model from json and weights files"""
    try:
        # Load the model architecture from JSON
        with open("sentiment_model.json", "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        
        # Load weights
        model.load_weights("Data_noiseNshift.h5")
        
        # Compile the model
        model.compile(loss='categorical_crossentropy', 
                     optimizer='adam', 
                     metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load the model at the start of your app
model = load_model()

# Check if model loaded successfully
if model is None:
    st.error("Failed to load model. Please check model files.")
    st.stop()

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
    """Predict emotion from preprocessed features and return emotion label and confidence"""
    try:
        # Define emotion labels (removing gender distinction)
        emotion_labels = {
            0: 'none',
            1: 'none',
            2: 'calm',
            3: 'happy',
            4: 'sad',
            5: 'angry',
            6: 'calm',
            7: 'happy',
            8: 'sad',
            9: 'angry',
            10: 'fearful',
            11: 'fearful'
        }
        
        # Get prediction
        prediction = model.predict(features, verbose=0)
        
        # Get the index of the highest probability
        predicted_class = np.argmax(prediction[0])
        
        # Get the confidence score
        confidence = prediction[0][predicted_class]
        
        # Get the emotion label
        emotion = emotion_labels[predicted_class]
        
        return {
            'emotion': emotion.capitalize(),
            'confidence': confidence,
            'raw_prediction': prediction[0]
        }
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def get_emotion_color(emotion_type):
    """Returns a color based on the emotion."""
    color_map = {
        'None': '#808080',    # Gray
        'Calm': '#98FB98',    # Pale Green
        'Happy': '#FFD700',   # Gold
        'Sad': '#4169E1',     # Royal Blue
        'Angry': '#FF4500',   # Red-Orange
        'Fearful': '#800080'  # Purple
    }
    return color_map.get(emotion_type, '#000000')

# File uploader widget
uploaded_file = st.file_uploader("Upload a .wav file", type=['wav'])

# When a file is uploaded
if uploaded_file is not None:
    # Play the audio file
    st.audio(uploaded_file, format='wav')
    
    # Read the audio file
    audio_data, sample_rate = sf.read(uploaded_file)
    
    # Preprocess and predict
    features = preprocess_audio(uploaded_file)
    if features is not None:
        result = predict_emotion(model, features)
        
        if result:
            # Display emotion with color coding
            emotion_color = get_emotion_color(result['emotion'])
            
            # Display emotion
            st.markdown(f"""
            <div style="
                padding: 20px;
                border-radius: 10px;
                background-color: {emotion_color};
                text-align: center;
                color: black;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 20px;
            ">
                {result['emotion']}
            </div>
            """, unsafe_allow_html=True)
            
            # Display confidence
            st.markdown(f"### Confidence: {result['confidence']*100:.2f}%")
            
            # Create a simplified bar chart of emotions
            unique_emotions = ['None', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful']
            
            # Aggregate probabilities for same emotions
            emotion_probs = {}
            for emotion in unique_emotions:
                emotion_lower = emotion.lower()
                # Sum probabilities for the same emotion regardless of gender
                mask = [e.lower().endswith(emotion_lower) for e in [
                    'female_none', 'male_none', 'male_calm', 'male_happy', 
                    'male_sad', 'male_angry', 'female_calm', 'female_happy',
                    'female_sad', 'female_angry', 'female_fearful', 'male_fearful'
                ]]
                emotion_probs[emotion] = sum(result['raw_prediction'][i] for i, m in enumerate(mask) if m)
            
            # Create DataFrame for chart
            chart_data = pd.DataFrame({
                'Emotion': list(emotion_probs.keys()),
                'Probability (%)': [v * 100 for v in emotion_probs.values()]
            })
            
            # Display bar chart
            st.markdown("### Probability Distribution")
            chart = alt.Chart(chart_data).mark_bar().encode(
                x='Probability (%)',
                y=alt.Y('Emotion', sort='-x'),
                color=alt.condition(
                    alt.datum['Probability (%)'] == max(chart_data['Probability (%)']),
                    alt.value('orange'),
                    alt.value('steelblue')
                )
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
