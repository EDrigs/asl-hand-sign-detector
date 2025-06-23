import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = "./data"


data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                # Collect all x, y, and z values for normalization
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                z_ = [lm.z for lm in hand_landmarks.landmark]

                # Normalize scale
                width = max(x_) - min(x_)
                height = max(y_) - min(y_)
                depth = max(z_) - min(z_)
                
                width = width if width != 0 else 1
                height = height if height != 0 else 1
                depth = depth if depth != 0 else 1

                # Use wrist as origin
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                wrist_z = hand_landmarks.landmark[0].z

                for lm in hand_landmarks.landmark:
                    x = (lm.x - wrist_x) / width
                    y = (lm.y - wrist_y) / height
                    z = (lm.z - wrist_z) / depth

                    # Append all 3 coordinates (optional: skip z if not needed)
                    data_aux.append(x)
                    data_aux.append(y)
                    data_aux.append(z)

            if len(data_aux) == 63:  # 21 landmarks × 3 values
                data.append(data_aux)
                labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
