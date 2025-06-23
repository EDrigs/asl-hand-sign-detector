import mediapipe as mp
import cv2
import numpy as np

class HandProcessor:
    def __init__(self):
        """
        Initialize the HandProcessor with the given hand object
        """
        self.mp_hand = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hand.Hands(static_image_mode=True, min_detection_confidence=0.3)


    def process_hand(self, frame):
        """
        Processes a given frame to extract hand landmarks.
        :param frame: The frame from the webcam (BGR Format)
        :return Processed landmarks or None if no hands are detected
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Iterate through all detected hands and extract landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmark connections on webcam
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hand.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                
                x_original = [lm.x for lm in hand_landmarks.landmark]
                y_original = [lm.y for lm in hand_landmarks.landmark]
                
                x_for_normalizing = x_original.copy()
                y_for_normalizing = y_original.copy()
                z_for_normalizing = [lm.z for lm in hand_landmarks.landmark]

                width = max(x_for_normalizing) - min(x_for_normalizing)
                height = max(y_for_normalizing) - min(y_for_normalizing)
                depth = max(z_for_normalizing) - min(z_for_normalizing)

                width = width if width != 0 else 1
                height = height if height != 0 else 1
                depth = depth if depth != 0 else 1

                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                wrist_z = hand_landmarks.landmark[0].z

                # Normalizing the data - scaling the landmark coordinates of the hand so that 
                # they are relative to the wrist and hand size rather than being relative to the webcam coordinates
                data_aux = [] # Auxiliary data. Temporary list used to store the processed hand landmark coordinates before returning them
                for lm in hand_landmarks.landmark:
                    x = (lm.x - wrist_x) / width
                    y = (lm.y - wrist_y) / height
                    z = (lm.z - wrist_z) / depth
                    # Append these values as three separate elements
                    data_aux.extend([x, y, z])

                if len(data_aux) == 63:
                    # Return the normalized data and the original x and y coordinates for drawing
                    return data_aux

        return None


    def reset_processor(self):
        """
        Reset the processor
        """
        self.hands.close()  # Close the current hand processor
        self.hands = self.mp_hand.Hands(
            static_image_mode=False, 
            min_detection_confidence=0.3, 
            min_tracking_confidence=0.3
        )





