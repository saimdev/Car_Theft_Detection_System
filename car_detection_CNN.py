import os
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

random.seed(42)

data_dir = './dataset'
categories = ['car', 'not_car']
img_size = 32 

def create_data():
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                img_array = cv2.resize(img_array, (img_size, img_size))
                data.append([img_array, class_num])
            except Exception as e:
                pass
    random.shuffle(data)
    X, y = [], []
    for features, label in data:
        X.append(features)
        y.append(label)
    X = np.array(X)
    y = to_categorical(y, num_classes=len(categories)) 
    return X, y

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(categories), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model()
    
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

    data_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                        height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)

    data_generator.fit(X_train)

    batch_size = 64
    epochs = 10
    model.fit(data_generator.flow(X_train, y_train, batch_size=batch_size), epochs=epochs,
              validation_data=(X_test, y_test), steps_per_epoch=len(X_train) // batch_size,callbacks=[tensorboard_callback])

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc}")

    model.save('car_detection_model.h5')

train_model()
