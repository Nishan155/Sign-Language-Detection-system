
import cv2
import pickle
import torch
import numpy as np
import os
import argparse
from detector import HandDetector, FeatureExtractor

# Optional: suppress TensorFlow/MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Dummy fallback MLP in case PyTorch model is used
class MLPClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLPClassifier, self).__init__()
        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(in_features, h))
            layers.append(torch.nn.ReLU())
            in_features = h
        layers.append(torch.nn.Linear(in_features, num_classes))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DigitRecognizer:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.hand_detector = HandDetector()
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.model_type = None
        self.scaler = None
        self.label_encoder = None
        self.load_model()

    def load_model(self):
        info_path = "/media/decoy/myssd/WEB/SLD/SLD_web/models/models/best_digits_model_info.pkl"
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"[ERROR] Info file not found: {info_path}")

        with open(info_path, "rb") as f:
            info = pickle.load(f)
        model_name = info.get("best_model_name")
        model_path = f"/media/decoy/myssd/WEB/SLD/SLD_web/models/models/{model_name}_digits_model.pkl"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERROR] Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.model_type = data["model_type"]
        self.scaler = data["scaler"]
        self.label_encoder = data["label_encoder"]

        if self.model_type == "pytorch_mlp":
            config = data["model_config"]
            self.model = MLPClassifier(config['input_size'], config['hidden_sizes'], config['num_classes'])
            self.model.load_state_dict(data["model_state_dict"])
            self.model.eval()
        else:
            self.model = data["model"]

    def predict(self, image):
        results = self.hand_detector.detect(image)
        features = self.feature_extractor.extract_features(results)

        if features is None:
            return None, 0.0

        scaled = self.scaler.transform([features])

        if self.model_type == "pytorch_mlp":
            with torch.no_grad():
                x = torch.tensor(scaled, dtype=torch.float32)
                output = self.model(x)
                probs = torch.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)
                label = self.label_encoder.inverse_transform([pred.item()])[0]
                return label, confidence.item()
        else:
            pred = self.model.predict(scaled)[0]
            label = self.label_encoder.inverse_transform([pred])[0]
            confidence = (
                np.max(self.model.predict_proba(scaled)) if hasattr(self.model, "predict_proba") else 1.0
            )
            return label, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image file")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"[ERROR] Image file not found: {args.image_path}")
        return

    image = cv2.imread(args.image_path)
    if image is None:
        print("[ERROR] Failed to read image.")
        return

    recognizer = DigitRecognizer()
    label, conf = recognizer.predict(image)

    if label:
        print(f"[RESULT] Predicted Digit: {label} | Confidence: {conf:.2f}")
    else:
        print("[INFO] No hand detected.")


if __name__ == "__main__":
    main()
