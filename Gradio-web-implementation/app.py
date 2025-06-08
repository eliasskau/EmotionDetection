import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import shutil
import time
from features import extract_features_consistent as extract_features  # ✅ Use shared function

# Define model
class EmotionNet3(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EmotionNet3, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Setup
INPUT_DIM = 38
NUM_CLASSES = 8
emotion_labels = ['Angry', 'Calm', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "emotion_model3.pt")
scaler_path = os.path.join(BASE_DIR, "feature_scaler.pkl")
user_data_dir = os.path.join(BASE_DIR, "user_data")
os.makedirs(user_data_dir, exist_ok=True)

# Load model and scaler
model = EmotionNet3(INPUT_DIM, NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()
scaler = joblib.load(scaler_path)

# Prediction + optional saving
def predict_and_optionally_save(audio_path, want_to_save, user_label):
    if audio_path is None or not os.path.isfile(audio_path):
        return "<div style='color:red;'>⚠️ No audio detected. Please try again.</div>"

    features = extract_features(audio_path)
    features_scaled = scaler.transform([features])
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(features_tensor)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        predicted_label = emotion_labels[predicted_idx]

    # Styled HTML result
    if want_to_save and user_label:
        timestamp = int(time.time())
        filename = f"{user_label}_{timestamp}.wav"
        save_path = os.path.join(user_data_dir, filename)
        shutil.copy(audio_path, save_path)

        return f"""
        <div style="border: 2px solid #000000; padding: 20px; border-radius: 10px; background-color: #666666; text-align: center;">
            <h2 style="color: #2e7d32;"> You feel: <strong>{predicted_label.upper()}</strong></h2>
            <p><em>This is the model’s prediction based on your recording.</em></p>
            <p>📝 Your label: <strong>{user_label}</strong></p>
            <p>💾 Recording saved for training. Thank you!</p>
        </div>
        """
    else:
        return f"""
        <div style="border: 2px solid #000000; padding: 20px; border-radius: 10px; background-color: #666666; text-align: center;">
            <h2 style="color: #0d47a1;"> You feel: <strong>{predicted_label.upper()}</strong></h2>
            <p><em>This is the model’s prediction based on your recording.</em></p>
            <p> This recording wasn’t saved. You’re just experimenting!</p>
        </div>
        """

# UI
def label_visible(save):
    return gr.update(visible=save)

with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Emotion Recognition Demo")
    gr.Markdown("Record your voice to see what emotion the model detects. Optionally, you can label your emotion and help us improve the model!")

    audio = gr.Audio(type="filepath", label="Record or upload your voice")
    save_opt = gr.Checkbox(label="📝 I want to label this audio and help improve the model!", value=False)
    label_dropdown = gr.Dropdown(choices=emotion_labels, label="Your emotion label", visible=False)
    save_opt.change(fn=label_visible, inputs=save_opt, outputs=label_dropdown)

    output = gr.HTML()
    btn = gr.Button("Submit")
    btn.click(fn=predict_and_optionally_save, inputs=[audio, save_opt, label_dropdown], outputs=output)

demo.launch(share=True)
