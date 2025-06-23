import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import cv2
import numpy as np
import time
from HandTrackingModule import HandTrackingModule as hd
from datetime import datetime

def save_to_path(image, path, sign, suffix=""):
    """
    Save the images to the specified path
    :param image: Image to save
    :param path: Path to save the images
    :param sign: Sign name to save the images with
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Check how many images already exist for this sign
    # existing_files = [f for f in os.listdir(path) if f.startswith(sign) and f.endswith(".png")]
                      

    # Have one number over the most recent to avoid overwriting
    # count = len(existing_files) + 1

    # Use timestamp for uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{sign}_{timestamp}{suffix}.png"

    cv2.imwrite(os.path.join(path, filename), image)
    # Extra print statement in case the programmer wants the file name that's saved
    # print(f"[SAVED] {filename}")



def collect_data(path, sign, interval=1):
    """
    Slightly modified version of capture_signs() to collect data for training the model
    :param path: Path to save the images
    :param sign: Sign name to save the images with
    :param interval: Interval in seconds to capture images
    """
    cap = cv2.VideoCapture(0)

    # ROI Coordinates
    x1, y1 = 50, 50
    x2, y2 = 350, 350

    # Initialize the hand detector
    detector = hd.handDetector()

    # Track frame count for the webcam video
    prev_capture_time = time.time()

    active = True
    saved_count = 0
    while active:
        # Original image
        success, img = cap.read()

        # Using this image to draw the rectangle
        img_copy = img.copy()

        img = detector.findHands(img)

        if detector.results.multi_hand_landmarks:

            hand_landmarks = detector.results.multi_hand_landmarks[0]

            h, w, _ = img.shape

            x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
            y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]

            # Add a margin around the hand
            margin = 20
            x_min = max(min(x_coords) - margin, 0)
            x_max = min(max(x_coords) + margin, w)

            y_min = max(min(y_coords) - margin, 0)
            y_max = min(max(y_coords) + margin, h)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Time check to save image
            current_time = time.time()
            if current_time - prev_capture_time >= interval:
                roi = img_copy[y_min:y_max, x_min:x_max]
                flipped = cv2.flip(roi, 1)
                if roi.size > 0 and roi.shape[0] > 10 and roi.shape[1] > 10:
                    save_to_path(roi, path, sign)
                    save_to_path(flipped, path, sign, suffix="_flipped")
                    saved_count += 2
                    print(f"Amount saved this session: {saved_count}")

                prev_capture_time = current_time
                cv2.imshow("Sign", roi)   
                                
        
        cv2.imshow("Hand Tracking", img)
        wait = cv2.waitKey(1)
        if wait == 27:  # Press 'ESC' to exit
            active = False

    cap.release()
    cv2.destroyAllWindows()






