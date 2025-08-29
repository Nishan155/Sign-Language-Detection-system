# test_alphabets.py
import cv2
import pickle
import torch
import numpy as np
import os
import argparse
from datetime import datetime
from detector import HandDetector, FeatureExtractor
from train import MLPClassifier


class AlphabetRecognizer:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.hand_detector = HandDetector()
        self.feature_extractor = FeatureExtractor()
        self.load_model()

    def load_model(self):
        info_path = os.path.join(self.model_dir, "best_alphabets_model_info.pkl")
        if not os.path.exists(info_path):
            raise FileNotFoundError("Train an alphabet model first.")

        with open(info_path, "rb") as f:
            info = pickle.load(f)
        model_name = info["best_model_name"]
        print(model_name)
        model_path = os.path.join(self.model_dir, f"{model_name}_alphabets_model.pkl")

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.model_type = data["model_type"]
        self.scaler = data["scaler"]
        self.label_encoder = data["label_encoder"]
        self.model_accuracy = data["accuracy"]

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

        scaled = self.scaler.transform(features.reshape(1, -1))
        if self.model_type == "pytorch_mlp":
            with torch.no_grad():
                x = torch.FloatTensor(scaled)
                out = self.model(x)
                probs = torch.softmax(out, dim=1)
                conf, pred = torch.max(probs, 1)
                label = self.label_encoder.inverse_transform([pred.item()])[0]
                return label, conf.item()
        else:
            pred = self.model.predict(scaled)[0]
            label = self.label_encoder.inverse_transform([pred])[0]
            conf = np.max(self.model.predict_proba(scaled)) if hasattr(self.model, "predict_proba") else 1.0
            return label, conf


def capture_image():
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture, ESC to cancel.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("Cancelled.")
            return None
        elif key == 32:
            filename = f"captured_alphabets/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            os.makedirs("captured_alphabets", exist_ok=True)
            cv2.imwrite(filename, frame)
            cap.release()
            cv2.destroyAllWindows()
            print(f"Image captured and saved to {filename}")
            return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", nargs="?", help="Path to image file (optional)")
    args = parser.parse_args()

    recognizer = AlphabetRecognizer()

    image = cv2.imread(args.image_path) if args.image_path else capture_image()
    if image is None:
        print("No image provided.")
        return

    letter, confidence = recognizer.predict(image)
    if letter:
        print(f"Predicted Letter: {letter}, Confidence: {confidence:.3f}")
    else:
        print("No hand detected.")


if __name__ == "__main__":
    main()
