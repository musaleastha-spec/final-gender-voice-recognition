import streamlit as st
import joblib
import librosa
import numpy as np

# Set page layout and title
st.set_page_config(page_title="Voice Gender Classifier", page_icon="🎙️", layout="centered")

st.title("🎙️ Voice Gender Recognition AI")
st.write("Upload a `.wav` file or record directly using your microphone to classify speaker gender.")

# 1. Load trained model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# 2. Define prediction logic
def classify_voice(audio_file):
    try:
        # Load audio signal (first 3 seconds) directly from Streamlit file buffer
        y, sr = librosa.load(audio_file, duration=3.0, res_type='kaiser_fast')
        
        # Extract MFCC features (20 coefficients: mean + std = 40 features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_features = np.hstack([np.mean(mfcc.T, axis=0), np.std(mfcc.T, axis=0)]).reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(mfcc_features)
        
        # Get output prediction or probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled_features)[0]
            classes = model.classes_
            return {str(classes[i]): float(probs[i]) for i in range(len(classes))}
        
        pred = model.predict(scaled_features)[0]
        return {str(pred): 1.0}
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# 3. Streamlit Input Widgets
tab1, tab2 = st.tabs(["📁 Upload File", "🎙️ Record Live"])

selected_audio = None

with tab1:
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
    if uploaded_file:
        selected_audio = uploaded_file

with tab2:
    recorded_file = st.audio_input("Record your voice")
    if recorded_file:
        selected_audio = recorded_file

# 4. Process Audio & Display Results
if selected_audio is not None:
    st.audio(selected_audio)
    
    if st.button("Analyze Voice", type="primary"):
        with st.spinner("Extracting spectral features..."):
            results = classify_voice(selected_audio)
            
            if results:
                st.subheader("Prediction Results:")
                for class_name, confidence in results.items():
                    col1, col2 = st.columns([1, 3])
                    col1.write(f"**{class_name.upper()}**")
                    col2.progress(confidence, text=f"{confidence * 100:.1f}%")
