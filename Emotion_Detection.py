import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load the face classifier and the emotion classifier
face_classifier = cv2.CascadeClassifier('C:\Complete-Working-Contact-Form-in-react-js-06-Modern-Contact-Form-React-through-EmailJs-main\Face_And_Emotion_Detection-master\Face_And_Emotion_Detection-master\Harcascade\haarcascade_frontalface_default.xml')
classifier = load_model('C:\Complete-Working-Contact-Form-in-react-js-06-Modern-Contact-Form-React-through-EmailJs-main\Face_And_Emotion_Detection-master\Face_And_Emotion_Detection-master\Models\model_v_47.hdf5')

# Class labels
class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    allfaces = []
    rects = []
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Normalize and reshape
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension
        roi_gray = np.expand_dims(roi_gray, axis=0)   # Add batch dimension
        
        preds = classifier.predict(roi_gray)[0]
        label = class_labels[preds.argmax()]
        
        label_position = (x + int(w/2), y - 10)
        cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Emotion Detector", img)

    if cv2.waitKey(1) == 13:  # Enter key to exit
        break

cap.release()
cv2.destroyAllWindows()
