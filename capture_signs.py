import os
import pickle
import mediapipe as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import cv2
import numpy as np
import time
from SignLanguageModel import SignLanguageModel
from HandProcessor import HandProcessor

def capture_signs():
    """
    Capture video from the webcam and detect ASL signs
    """
    cap = cv2.VideoCapture(0)

    # Hand Processor
    processor = HandProcessor()

    # Load model
    model_path = "landmarks_model.joblib"
    model = SignLanguageModel(model_path)

    # Keep a timer that will judge when to predict a sign
    prev_capture_time = time.time()
    margin = 20

    active = True

    # Create an AccessControl class with a authorized password as a parameter
    # In this test case, we will choose "yang" for Dr. Yang Song :)
    while active:
        # Original image
        success, img = cap.read()

        hand_data = processor.process_hand(img)

        if hand_data is not None and len(hand_data) == 63:

            # Time check to save image (optional)
            current_time = time.time()
            if current_time - prev_capture_time >= 1:  # Save every 1 second

                prediction = model.identify_sign([np.asarray(hand_data)])

                print(prediction)
                                
                prev_capture_time = current_time
        
        cv2.imshow("Hand Tracking", img)
        wait = cv2.waitKey(1)

        if wait == 27:  # Press 'ESC' to exit
            active = False

    cap.release()
    cv2.destroyAllWindows()


def main():
    capture_signs()

if __name__ == "__main__":
    main()