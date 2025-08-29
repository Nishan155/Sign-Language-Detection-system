import torch
import torch.nn as nn
import numpy as np
import json
import time
import cv2
import os
import sys
from feature import SignLanguageFeatureExtractor

# === Model definition (must match training!) ===
class ImprovedSignLanguageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(ImprovedSignLanguageLSTM, self).__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=(dropout if num_layers > 1 else 0), bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        out = self.dropout(attended_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# === Feature processing utility ===
def process_features_static(features_sequence):
    numerical_sequence = []
    for frame_features in features_sequence:
        frame_vector = []
        for hand_key in ['left_hand', 'right_hand']:
            if hand_key in frame_features:
                hand_data = frame_features[hand_key]
                for rel_key in ['relative_to_face', 'relative_to_nose', 'relative_to_mouth',
                                'relative_to_ear', 'relative_to_chest']:
                    frame_vector.extend(hand_data.get(rel_key, [0.0, 0.0]))
                for point_key in ['wrist', 'index_tip', 'thumb_tip']:
                    frame_vector.extend(hand_data.get(point_key, [0.0, 0.0]))
                frame_vector.append(1.0)
            else:
                frame_vector.extend([0.0] * 16 + [0.0])
        numerical_sequence.append(frame_vector)
    return numerical_sequence

# === Feature extraction from video file ===
def extract_features_from_video(video_path, extractor):
    """Extract features from a pre-recorded video file."""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None
    
    print(f"📹 Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video file: {video_path}")
        return None
    
    features_sequence = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract features from current frame
        frame_features = extractor.extract_features(frame)
        if frame_features:
            features_sequence.append(frame_features)
        
        frame_count += 1
        
        # Optional: Display progress
        if frame_count % 10 == 0:
            print(f"   Processed {frame_count} frames...", end='\r')
    
    cap.release()
    print(f"\n✅ Processed {frame_count} frames from video")
    
    if not features_sequence:
        print("❌ No valid features extracted from video")
        return None
    
    return features_sequence

# === Prediction pipeline ===
def predict_gloss(feature_sequence, model, scaler, label_encoder, max_seq_len=30):
    numerical_sequence = process_features_static(feature_sequence)
    if len(numerical_sequence) == 0:
        print("❌ No valid hand data detected.")
        return None, 0.0

    norm_sequence = scaler.transform(numerical_sequence)

    # Pad/truncate
    if len(norm_sequence) < max_seq_len:
        pad = np.tile(norm_sequence[-1], (max_seq_len - len(norm_sequence), 1))
        norm_sequence = np.vstack([norm_sequence, pad])
    else:
        norm_sequence = norm_sequence[:max_seq_len]

    input_tensor = torch.FloatTensor(norm_sequence).unsqueeze(0)  # shape: (1, seq_len, features)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()

    predicted_gloss = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_gloss, confidence

# === Main test logic ===
def main():
    # Check command line arguments
    video_path = None
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"Video mode: Using video file '{video_path}'")
    else:
        print("Real-time mode: Will capture from camera")
    
    # Load model checkpoint
    checkpoint_path = '/media/decoy/myssd/WEB/SLD/SLD_web/test/sign_language_model.pth'
    if not os.path.exists(checkpoint_path):
        print("Model not found. Please train the model first.")
        return

    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = ImprovedSignLanguageLSTM(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    scaler = checkpoint['scaler']
    label_encoder = checkpoint['label_encoder']
    max_seq_len = checkpoint.get('max_seq_len', 30)
    
    print("Model loaded successfully!")
    print(f"Model info: {checkpoint['num_classes']} classes, max sequence length: {max_seq_len}")

    # Initialize extractor
    extractor = SignLanguageFeatureExtractor()

    # Get features based on mode
    if video_path:
        # Video file mode
        features_sequence = extract_features_from_video(video_path, extractor)
        if features_sequence is None:
            return
    print("Making prediction...")
    predicted_gloss, confidence = predict_gloss(features_sequence, model, scaler, label_encoder, max_seq_len)

    if predicted_gloss is not None:
        print(f"\nPredicted Sign: **{predicted_gloss}** (Confidence: {confidence:.2%})")


if __name__ == '__main__':
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        sys.exit(0)
    
    main()