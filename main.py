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
from AccessControl import AccessControl

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

    password = input("\nProvide a password you would like to use for access control: ")

    # Create an AccessControl class with a authorized password as a parameter
    validator = AccessControl(password)

    while active:

        # Original image
        success, img = cap.read()

        hand_data = processor.process_hand(img)

        if hand_data is not None and len(hand_data) == 63:

            # Time check to save image (optional)
            current_time = time.time()
            if current_time - prev_capture_time >= 2:  # Save every 1 second

                prediction = model.identify_sign([np.asarray(hand_data)])
                                
                prev_capture_time = current_time

                # Add the inputted sign to the password "total"
                validator.build_input_password(prediction)
                print(prediction, end="", flush=True)
        
        cv2.imshow("Hand Tracking", img)
        wait = cv2.waitKey(1)
        if wait == 13: # Enter key (ASCII 13)
            if validator.is_authorized():
                validator.grant_access()
                active = False
            else:
                validator.deny_access()
                validator.clear_current_input() # Erase the user password input so that they may start over
                print("Please try again, or click ESC to quit.")

        elif wait == 27:  # Press 'ESC' to exit
            active = False

    cap.release()
    cv2.destroyAllWindows()


def main():
    capture_signs()

if __name__ == "__main__":
    main()