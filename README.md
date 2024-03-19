# Real-time Sign Language Recognition

Welcome to the Real-time Sign Language Recognition project! This project aims to recognize American Sign Language (ASL) alphabets in real-time using computer vision techniques and machine learning. Below are the details on how to run the project locally and an overview of its components:

## Dataset

The dataset used for training and testing the model is available on Kaggle at the following link: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). It contains images of ASL alphabet signs.

## Data Processing

To convert the images into numeric data, you can run the [`create_dataset.py`](https://github.com/adiren7/Real_time_sign_language_recognition/blob/main/create_dataset.py) script. This script utilizes the Mediapipe framework to extract hand landmarks from the images, which are then saved in pickle format and used as features for training the model.

## Model Training

For predicting alphabet classes, a Random Forest model from the Scikit-learn library is used. The [`train_classifier.py`](https://github.com/adiren7/Real_time_sign_language_recognition/blob/main/train_classifier.py) script loads the numeric data, trains the Random Forest model, and saves it in pickle format for later use.

## Real-time Recognition

To demonstrate real-time recognition, the project uses OpenCV and Mediapipe. OpenCV is used for accessing the webcam and displaying video streams, while Mediapipe is utilized for detecting hand landmarks in real-time. The trained Random Forest model predicts the alphabet based on the detected hand landmarks, and the results are displayed on the screen.

## Getting Started

To run the Real-time Sign Language Recognition system locally using your dataset and model weights, follow these steps:

1. **Clone the Repository**:
```
git clone https://github.com/adiren7/Real_time_sign_language_recognition.git
cd Real-time-Sign-Language-Recognition
```

2. **Install Dependencies**:
```
pip install -r requirements.txt
```

3. **Run the Application**:
```
python sign_detection.py
```

## Example Usage

Here's an example of how to use the Real-time Sign Language Recognition system:

1. Run the [`sign_detection.py`](https://github.com/adiren7/Real_time_sign_language_recognition/blob/main/sign_detection.py) script.
2. Position your hand in front of the webcam.
3. The system will detect your hand landmarks and predict the corresponding ASL alphabet.
4. The predicted alphabet will be displayed on the screen in real-time.

## Contributions

Contributions to this project are welcome! If you have any ideas for improvements or feature suggestions, feel free to open an issue or submit a pull request.



## Acknowledgements

- Special thanks to Kaggle user `grassknoted` for providing the ASL Alphabet Dataset.
