import tensorflow as tf
import sys
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datastuff = ImageDataGenerator(rescale=1./255)

test_ds = datastuff.flow_from_directory(
        sys.argv[1],
        shuffle=False,
        class_mode="binary") 

model = keras.models.load_model(sys.argv[2])

results = model.evaluate(test_ds)

print("test err:", 1-results[1])
