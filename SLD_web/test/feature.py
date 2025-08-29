import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time

class SignLanguageFeatureExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def extract_features(self, frame):
        """Extract hand positions relative to face and body landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get detections
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        features = {}
        
        # Reference points
        face_center = None
        nose_tip = None
        mouth_center = None
        left_ear = None
        right_ear = None
        chest_center = None
        
        # Extract face landmarks
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            # Nose tip (landmark 1)
            nose_tip = (face_landmarks.landmark[1].x, face_landmarks.landmark[1].y)
            # Mouth center (landmarks 13, 14)
            mouth_center = (
                (face_landmarks.landmark[13].x + face_landmarks.landmark[14].x) / 2,
                (face_landmarks.landmark[13].y + face_landmarks.landmark[14].y) / 2
            )
            # Face center approximation
            face_center = (
                sum([lm.x for lm in face_landmarks.landmark]) / len(face_landmarks.landmark),
                sum([lm.y for lm in face_landmarks.landmark]) / len(face_landmarks.landmark)
            )
        
        # Extract pose landmarks
        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks.landmark
            # Ears
            left_ear = (pose_landmarks[7].x, pose_landmarks[7].y)
            right_ear = (pose_landmarks[8].x, pose_landmarks[8].y)
            # Chest center (shoulders midpoint)
            chest_center = (
                (pose_landmarks[11].x + pose_landmarks[12].x) / 2,
                (pose_landmarks[11].y + pose_landmarks[12].y) / 2
            )
        
        # Extract hand features
        if hand_results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                hand_label = hand_results.multi_handedness[i].classification[0].label
                
                # Key hand points
                wrist = (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y)
                index_tip = (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y)
                thumb_tip = (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y)
                
                hand_features = {
                    'wrist': wrist,
                    'index_tip': index_tip,
                    'thumb_tip': thumb_tip,
                    'palm_landmarks': [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                }
                
                # Calculate relative positions
                if face_center:
                    hand_features['relative_to_face'] = (
                        wrist[0] - face_center[0],
                        wrist[1] - face_center[1]
                    )
                
                if nose_tip:
                    hand_features['relative_to_nose'] = (
                        wrist[0] - nose_tip[0],
                        wrist[1] - nose_tip[1]
                    )
                
                if mouth_center:
                    hand_features['relative_to_mouth'] = (
                        wrist[0] - mouth_center[0],
                        wrist[1] - mouth_center[1]
                    )
                
                if left_ear and right_ear:
                    ear_center = ((left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2)
                    hand_features['relative_to_ear'] = (
                        wrist[0] - ear_center[0],
                        wrist[1] - ear_center[1]
                    )
                
                if chest_center:
                    hand_features['relative_to_chest'] = (
                        wrist[0] - chest_center[0],
                        wrist[1] - chest_center[1]
                    )
                
                features[f'{hand_label.lower()}_hand'] = hand_features
        
        return features
    
    def capture_and_extract(self, gloss, video_id, duration=2):
        """Capture video and extract features"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return None
        
        print(f"Recording {gloss} - Video {video_id} in 3 seconds...")
        time.sleep(3)
        
        frames = []
        features_sequence = []
        
        start_time = time.time()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        print(f"Recording for {duration} seconds...")
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract features from current frame
            features = self.extract_features(frame)
            features_sequence.append(features)
            
            # Show frame (optional)
            cv2.imshow('Recording', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save features
        os.makedirs(f'data/{gloss}', exist_ok=True)
        
        output_data = {
            'gloss': gloss,
            'video_id': video_id,
            'features_sequence': features_sequence,
            'num_frames': len(features_sequence),
            'duration': duration
        }
        
        filename = f'data/{gloss}/video_{video_id}Features.json'
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Features saved to {filename}")
        return output_data

def main():
    extractor = SignLanguageFeatureExtractor()
    
    while True:
        gloss = input("Enter sign gloss (or 'quit' to exit): ").strip()
        if gloss.lower() == 'quit':
            break
        
        if not gloss:
            continue
        
        # Capture 10 samples
        for i in range(10):
            input(f"Press Enter to record sample {i+1}/10 for '{gloss}'...")
            extractor.capture_and_extract(gloss, i+1)
        
        print(f"Completed all 10 samples for '{gloss}'\n")

if __name__ == "__main__":
    main()