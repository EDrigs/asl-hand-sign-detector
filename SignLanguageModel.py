import joblib
import numpy as np

class SignLanguageModel:
    def __init__(self, model_path):
        """
        Initialize the SignLanguageModel with the given path to the model file.
        :param model_path: Path to the model file
        """
        self.model = joblib.load(model_path)

    def identify_sign(self, input_data):
        """
        Predict the sign language character from the input data
        :param input_data: Preprocessed image data
        """
        prediction = self.model.predict(input_data)
        return prediction[0]
