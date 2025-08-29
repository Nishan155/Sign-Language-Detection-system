import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

class HandDetector:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.5
        )
        self.drawer = mp.solutions.drawing_utils

    def detect(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def draw(self, img, results):
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.drawer.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)
        return img

class FeatureExtractor:
    def extract_features(self, results):
        if not results.multi_hand_landmarks:
            return None

        features = []
        for hand_landmarks in results.multi_hand_landmarks:
            coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            min_vals = coords.min(axis=0)
            max_vals = coords.max(axis=0)
            normalized = (coords - min_vals) / (max_vals - min_vals + 1e-6)
            flattened = normalized.flatten().tolist()

            # Orientation angle (wrist to middle fingertip)
            wrist = coords[0]
            middle_tip = coords[12]
            angle = np.arctan2(middle_tip[1] - wrist[1], middle_tip[0] - wrist[0])
            flattened += [np.cos(angle), np.sin(angle)]

            # Pairwise distances between select landmark pairs
            pair_indices = [(4, 8), (4, 12), (4, 16), (4, 20), (8, 12), (8, 16), (8, 20)]
            for i, j in pair_indices:
                dist = np.linalg.norm(normalized[i] - normalized[j])
                flattened.append(dist)

            features.extend(flattened)

        return np.array(features, dtype=np.float32)

class DataCapture:
    def __init__(self, save_dir="asl_data"):
        self.detector = HandDetector()
        self.extractor = FeatureExtractor()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def capture(self, label_list, data_type="data"):
        features, labels = [], []
        cap = cv2.VideoCapture(0)
        samples_per_class = 60

        print(f"\nCapturing {data_type.upper()} data. Press 's' to save sample, 'n' for next label, 'q' to quit.\n")
        idx, count = 0, 0
        hand_stage = 0  # 0: left hand, 1: right hand

        while idx < len(label_list):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            results = self.detector.detect(frame)
            self.detector.draw(frame, results)

            label = label_list[idx]
            hand_label = "Left" if hand_stage == 0 else "Right"
            cv2.putText(frame, f"{label} [{hand_label}] ({count}/{samples_per_class})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
            cv2.imshow('Capture', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                feats = self.extractor.extract_features(results)
                if feats is not None and len(results.multi_hand_landmarks) == 1:
                    label_full = f"{label}_{hand_label.lower()}"
                    features.append(feats)
                    labels.append(label_full)
                    count += 1
                    print(f"Saved {label_full}: {count}")
                    if count >= samples_per_class:
                        count = 0
                        if hand_stage == 0:
                            hand_stage = 1
                        else:
                            hand_stage = 0
                            idx += 1
                else:
                    print("Please show only one hand clearly.")
            elif key == ord('n'):
                hand_stage = 0
                count = 0
                idx += 1
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self._save(features, labels, label_list, data_type)

    def _save(self, features, labels, label_names, name):
        if features:
            data = {
                'features': np.array(features),
                'labels': labels,
                'label_names': label_names
            }
            filepath = os.path.join(self.save_dir, f"{name}_data.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            print(f"\nSaved to {filepath}. Total samples: {len(features)}")
        else:
            print("No data collected.")

def main():
    capture = DataCapture()
    print("ASL Capture\n1. Digits\n2. Alphabets\n3. Exit")

    while True:
        choice = input("\nChoose (1-3): ")
        if choice == '1':
            capture.capture([str(i) for i in range(10)], "digits")
        elif choice == '2':
            capture.capture([chr(i) for i in range(65, 91)], "alphabets")
        elif choice == '3':
            break
        else:
            print("Invalid input.")

if __name__ == "__main__":
    main()
