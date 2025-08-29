import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class ASLTrainer:
    def __init__(self, data_dir="asl_data", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_data(self, data_type="digits"):
        path = os.path.join(self.data_dir, f"{data_type}_data.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"[INFO] Loaded {data_type} data: {data['features'].shape}, Classes: {len(set(data['labels']))}")
        return data

    def preprocess_data(self, features, labels):
        X_scaled = self.scaler.fit_transform(features)
        y_encoded = self.label_encoder.fit_transform(labels)
        return X_scaled, y_encoded

    def train_model(self, model_type="random_forest", data_type="digits", test_size=0.2):
        print(f"\n=== Training {model_type.upper()} for: {data_type.upper()} ===")
        data = self.load_data(data_type)
        X, y = self.preprocess_data(data["features"], data["labels"])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                max_iter=300,
                random_state=42
            )
        else:
            raise ValueError("Unsupported model type. Use 'random_forest' or 'mlp'.")

        print("[INFO] Training model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"[RESULT] Train Accuracy: {train_acc:.4f}")
        print(f"[RESULT] Test Accuracy:  {test_acc:.4f}\n")

        print("[INFO] Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        # Save model
        model_filename = f"{model_type}_{data_type}_model.pkl"
        model_path = os.path.join(self.model_dir, model_filename)
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": model,
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "model_type": model_type,
                "accuracy": test_acc,
                "data_type": data_type
            }, f)

        print(f"[INFO] Model saved to {model_path}")

        # Save best model info
        info_path = os.path.join(self.model_dir, f"best_{data_type}_model_info.pkl")
        with open(info_path, "wb") as f:
            pickle.dump({
                "best_model_name": model_type,
                "accuracy": test_acc,
                "data_type": data_type
            }, f)
        print(f"[INFO] Model info saved to {info_path}")

        self.plot_confusion_matrix(y_test, y_pred, self.label_encoder.classes_, title=f"{model_type.upper()} - {data_type.title()}")

    def plot_confusion_matrix(self, y_true, y_pred, classes, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        filename = f"confusion_matrix_{title.lower().replace(' ', '_')}.png"
        path = os.path.join(self.model_dir, filename)
        plt.savefig(path)
        plt.show()
        print(f"[INFO] Confusion matrix saved as {path}")


def cli_interface():
    trainer = ASLTrainer()
    print("\nASL Model Trainer\n")

    while True:
        print("\n1. Train Random Forest - Digits")
        print("2. Train Random Forest - Alphabets")
        print("3. Train MLP - Digits")
        print("4. Train MLP - Alphabets")
        print("5. Exit")
        choice = input("Enter choice (1-5): ")

        if choice == '1':
            trainer.train_model(model_type="random_forest", data_type="digits")
        elif choice == '2':
            trainer.train_model(model_type="random_forest", data_type="alphabets")
        elif choice == '3':
            trainer.train_model(model_type="mlp", data_type="digits")
        elif choice == '4':
            trainer.train_model(model_type="mlp", data_type="alphabets")
        elif choice == '5':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Try again.")


def main():
    parser = argparse.ArgumentParser(description="ASL Model Trainer")
    parser.add_argument("--type", choices=["digits", "alphabets"], help="Dataset type to train on")
    parser.add_argument("--model", choices=["random_forest", "mlp"], help="Model type to train")
    args = parser.parse_args()

    if args.type and args.model:
        trainer = ASLTrainer()
        trainer.train_model(model_type=args.model, data_type=args.type)
    else:
        cli_interface()


if __name__ == "__main__":
    main()
