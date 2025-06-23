# Using ASL Hand Signs for Access Control with Computer Vision and an ASL-trained model
## By Ethan Driggers
### CSC-592: Computer Vision

#### 📂 Main Program Files:
`AccessControl.py`: Class to handle access control functionality during webcam capture. 

`collect_data.py`: Script used to collect images for each sign.

`main.py`: Driver file to run for the ASL webcam capture.

`HandProcessor.py`: Class to process a hand when it appears within the webcam. Utilizes the `mediapipe` library to identify and draw hand landmarks.

`modify_data.py`: Script originally created by [computervisioneng](https://github.com/computervisioneng/sign-language-detector-python) to collect hand landmark information from image datasets of certain handsigns. Modified to fit the needs of my project. Converts data to a `.pickle` file

`SignLanguageModel,py`: Class to load AI model and use it for sign prediction.

`TrainingASLWebcamModel.ipynb`: Notebook utilized to train and save a model on information from the `.pickle` file.

`capture_signs.py`: Model prediction demo webcam that can be used to see the model's ability to predict signs.

#### 📁 Other files/folders

`landmarks_model.joblib`: Model trained on the ASL hand sign landmark data

`data.pickle`: File to train a model on. Contains a dictionary with labels to train the model with.

`archive`: Directory containing unsuccessful model building and data collection attempts. Models have the potential to be useful in other contexts, but here, they contained large **generalization gaps**.

`data`: Dataset for training. Over 7,000 images containing ASL hand signs A-Y.

`report`: Reports written during project development

#### 🛠️ How to Use:
1. Open the program in a code editor of your choice
1. Go to `main.py`
1. Run `main.py`. It may take several seconds on the initial runtime for your machine.
1. Input a password you would like to use to test the access control.
1. Use the necessary sign and allow the model to predict the sign so that you may "type" your password. If you are not familiar with ASL alphabet hand signs, images of what they look like can be found [here](https://teachbesideme.com/asl-alphabet-printable-chart-and-flashcards/)
1. Alternatively, you can run the `capture_signs.py` file just to see the model's prowess in prediction

### Thank you for using my project!