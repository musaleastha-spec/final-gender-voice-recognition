import gradio as gr
import joblib
import librosa
import numpy as np

# 1. Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# 2. Prediction logic
def classify_voice(audio_path):
    if audio_path is None:
        return "No audio detected."
    
    try:
        # Load audio signal (3-second duration)
        y, sr = librosa.load(audio_path, duration=3.0, res_type='kaiser_fast')
        
        # Extract MFCC features (20 coefficients: mean + std = 40 features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_features = np.hstack([np.mean(mfcc.T, axis=0), np.std(mfcc.T, axis=0)]).reshape(1, -1)
        
        # Scale extracted features
        scaled_features = scaler.transform(mfcc_features)
        
        # Output probabilities if supported by the model
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled_features)[0]
            classes = model.classes_
            return {str(classes[i]): float(probs[i]) for i in range(len(classes))}
        
        # Fallback for non-probabilistic models
        pred = model.predict(scaled_features)[0]
        return {str(pred): 1.0}
        
    except Exception as e:
        return f"Error processing audio: {str(e)}"

# 3. Build Gradio Interface
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme, title="Voice Gender Classifier") as demo:
    gr.Markdown(
        """
        # 🎙️ Voice Gender Recognition AI
        Upload a `.wav` file or record live using your microphone to classify speaker gender.
        """
    )
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio Input"
            )
            analyze_btn = gr.Button("Analyze Voice", variant="primary")
            
        with gr.Column():
            output_label = gr.Label(
                label="Prediction Confidence", 
                num_top_classes=2
            )
    
    analyze_btn.click(
        fn=classify_voice, 
        inputs=audio_input, 
        outputs=output_label,
        api_name="predict"  # Automatic API endpoint at /api/predict
    )

if __name__ == "__main__":
    demo.launch()
