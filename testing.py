import cv2
import numpy as np
from keras.models import load_model

def preprocess_image(img_path, img_size):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = np.expand_dims(img, axis=0)
    return img

def predict_car(img_path, model_path):
    img_size = 32 
    model = load_model(model_path)
    img = preprocess_image(img_path, img_size)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction[0])
    class_label = 'car' if class_idx == 0 else 'not a car'
    confidence = prediction[0][class_idx] * 100
    return class_label, confidence

model_path = 'car_detection_model.h5'
image_path = 'testing3.jpg'  

class_label, confidence = predict_car(image_path, model_path)

print(f"The image is {class_label} with a confidence of {confidence:.2f}%.")
