import os
import sys
import json
import torch
import numpy as np
import cv2
import mediapipe as mp
import torch.nn as nn
import torch.nn.functional as F
import warnings
import contextlib
from io import StringIO

# Suppress all warnings and logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

# Context manager to suppress all output
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# ==================== Import model class ====================
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.downsample:
            x = self.downsample(x)
        return self.relu(out + x)

class TCNClassifier(nn.Module):
    def __init__(self, input_size=126, num_classes=150):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(input_size, 128),
            TCNBlock(128, 256),
            TCNBlock(256, 128)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ==================== Configuration ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_FRAMES = 16
mp_hands = mp.solutions.hands

# ==================== Gloss Labels ====================
def load_gloss_labels(json_path="gloss.json"):
    try:
        with open(json_path, "r") as f:
            gloss_dict = json.load(f)
        return [gloss_dict[str(i)] for i in range(len(gloss_dict))]
    except:
        return None

# ==================== Load Trained Model ====================
def load_model(model_path="best_hand_tcn.pth", input_size=126, num_classes=None):
    try:
        model = TCNClassifier(input_size=input_size, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except:
        return None

# ==================== Hand Feature Extraction ====================
def extract_hand_landmarks_from_frame(frame, hands_detector):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return np.zeros(21 * 3 * 2, dtype=np.float32)

        hand_features = []
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            hand_features.append(landmarks)

        if len(hand_features) == 1:
            hand_features.append([0.0] * 21 * 3)

        return np.array(hand_features).flatten()
    except:
        return np.zeros(21 * 3 * 2, dtype=np.float32)

def extract_video_hand_features(video_path, target_frames=TARGET_FRAMES):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return None

        total_frames = len(frames)
        indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
        sampled_frames = [frames[i] for i in indices]

        features = []
        # Suppress MediaPipe output
        with suppress_output():
            with mp_hands.Hands(static_image_mode=True,
                                max_num_hands=2,
                                min_detection_confidence=0.5) as hands_detector:
                for frame in sampled_frames:
                    feat = extract_hand_landmarks_from_frame(frame, hands_detector)
                    features.append(feat)

        return np.array(features, dtype=np.float32)
    except:
        return None

# ==================== Prediction ====================
def predict_gloss(model, feature_tensor, gloss_labels):
    try:
        x = feature_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            pred_idx = logits.argmax(dim=1).item()
        return gloss_labels[pred_idx]
    except:
        return "ERROR"

# ==================== Main Function for Web ====================
def predict_video_gloss(video_path, gloss_json_path="/media/decoy/myssd/WEB/SLD/SLD_web/test/gloss.json", 
                        model_path="/media/decoy/myssd/WEB/SLD/SLD_web/test/best_hand_tcn.pth"):
    """
    Main function for web integration - returns only the predicted gloss or error message
    """
    # Validate input
    if not os.path.exists(video_path):
        return "ERROR: Video file not found"
    
    if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return "ERROR: Unsupported video format"
    
    # Load resources with suppressed output
    with suppress_output():
        gloss_labels = load_gloss_labels(gloss_json_path)
        if gloss_labels is None:
            return "ERROR: Failed to load gloss labels"
        
        model = load_model(model_path=model_path, input_size=126, num_classes=len(gloss_labels))
        if model is None:
            return "ERROR: Failed to load model"
        
        # Extract features
        features = extract_video_hand_features(video_path, TARGET_FRAMES)
        if features is None:
            return "ERROR: Feature extraction failed"
        
        if features.shape != (TARGET_FRAMES, 126):
            return "ERROR: Invalid feature dimensions"
        
        # Make prediction
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        prediction = predict_gloss(model, feature_tensor, gloss_labels)
    
    return prediction

# ==================== Web API Function ====================
def web_predict(video_path):
    """
    Simple function for web API - returns only the gloss prediction with all output suppressed
    """
    with suppress_output():
        result = predict_video_gloss(video_path)
    return result

# ==================== Main for Command Line ====================
def main(video_path):
    # For command line, we want to see the result but suppress MediaPipe logs
    with suppress_output():
        result = predict_video_gloss(video_path)
    print(result)  # Only print the final result

# ==================== Run ====================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: Usage: python test.py <path_to_video.mp4>")
        sys.exit(1)
    main(sys.argv[1])