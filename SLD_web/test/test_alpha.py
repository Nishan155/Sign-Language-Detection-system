import cv2
import pickle
import torch
import numpy as np
import os
import argparse
from detector import HandDetector, FeatureExtractor
from train import MLPClassifier


class AlphabetRecognizer:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.hand_detector = HandDetector()
        self.feature_extractor = FeatureExtractor()
        self.load_model()

    def load_model(self):
        info_path = "/media/decoy/myssd/WEB/SLD/SLD_web/models/models/best_alphabets_model_info.pkl"
        if not os.path.exists(info_path):
            raise FileNotFoundError("Train an alphabet model first.")

        with open(info_path, "rb") as f:
            info = pickle.load(f)
        model_name = info["best_model_name"]
        model_path = f"/media/decoy/myssd/WEB/SLD/SLD_web/models/models/random_forest_alphabets_model.pkl"

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
        print("\n[DEBUG] Scaled Input Features:")
        print(scaled.flatten())

        if self.model_type == "pytorch_mlp":
            with torch.no_grad():
                x = torch.FloatTensor(scaled)
                out = self.model(x)
                probs = torch.softmax(out, dim=1)
                conf, pred = torch.max(probs, 1)
                label = self.label_encoder.inverse_transform([pred.item()])[0]
                print(f"[INFO] Predicted using PyTorch MLP")
                return label, conf.item()
        else:
            pred = self.model.predict(scaled)[0]
            label = self.label_encoder.inverse_transform([pred])[0]
            probas = self.model.predict_proba(scaled)[0]
            conf = np.max(probas) if hasattr(self.model, "predict_proba") else 1.0
            print(f"[INFO] Predicted using {self.model_type}")
            print(f"[INFO] Class Probabilities: {probas}")

            # Optional: show top contributing features (for tree-based models)
            if hasattr(self.model, "feature_importances_"):
                top_features = np.argsort(self.model.feature_importances_)[::-1][:5]
                print("Reasoning:\n Top contributing features (index: importance):")
                for idx in top_features:
                    print(f"  Feature {idx}: {self.model.feature_importances_[idx]:.4f}")

            return label, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", nargs="?", help="Path to image file (optional)")
    args = parser.parse_args()

    recognizer = AlphabetRecognizer()

    image = cv2.imread(args.image_path)
    if image is None:
        print("No image provided.")
        return

    letter, confidence = recognizer.predict(image)
    if letter:
        print(f"\n[RESULT] Predicted Letter: {letter}, Confidence: {confidence:.3f}")
    else:
        print("[RESULT] No hand detected.")


if __name__ == "__main__":
    main()
