import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
model = tf.keras.models.load_model(filepath = "model.h5")

import numpy as np

class_names= ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50

def detect():
    # dataset = tf.keras.preprocessing.image_dataset_from_directory(
    #     "potato",
    #     shuffle = True,
    #     image_size = (IMAGE_SIZE, IMAGE_SIZE),
    #     batch_size = BATCH_SIZE)

    # for images, labels in dataset:
    #     print(type(images))
    img = tf.keras.preprocessing.image.load_img('potato/user.jpg')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)

    predictions = model.predict(img_array)
    print("Prediction is", predictions)
    print("this",np.argmax(predictions[0]))
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])),2)
    
    print(predicted_class)
    print(confidence)
    return predicted_class, confidence