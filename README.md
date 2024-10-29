Face Recognition and Emotion Detection
Project Overview
This project focuses on detecting human emotions through facial expressions using a Convolutional Neural Network (CNN) classifier. Leveraging the FER-2013 dataset, the model is trained to recognize various emotions, including happiness, sadness, surprise, anger, and more.

Key Features
Dataset: The model is trained on the FER-2013 dataset, which consists of 35,887 grayscale images of faces, labeled with seven different emotions.

Model Architecture:

Utilized Keras with a TensorFlow backend to build and train the CNN classifier.
Achieved a maximum accuracy of 60.1% during training, demonstrating the model's ability to generalize emotion recognition from facial expressions.
Face Detection:

Integrated OpenCV to detect faces within images. The face detection process ensures that only relevant facial data is passed to the classifier, improving the accuracy of emotion predictions.
Emotion Prediction:

Once a face is detected, the cropped facial image is processed and fed into the trained CNN model to predict the emotion of the individual.
Technical Stack
Programming Languages: Python
Libraries Used:
Keras: For building and training the CNN model.
TensorFlow: As the backend for Keras, enabling efficient computation.
OpenCV: For face detection in images.
NumPy: For numerical operations on image data.
