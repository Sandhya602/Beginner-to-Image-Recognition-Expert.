# main.py - Vision AI in 5 Days Project
# Author: Sandhya Shree B V
# Date: 2025-08-13

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Load and Preprocess Dataset (MNIST Example)
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape & normalize
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# 2. Build & Train CNN Model
print("Building CNN model...")
cnn_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = cnn_model.fit(
    x_train, y_train_cat,
    validation_data=(x_test, y_test_cat),
    epochs=5,
    batch_size=64
)

# Save training curves
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('CNN Training Accuracy')
plt.savefig('cnn_accuracy.png')
plt.close()

# 3. Data Augmentation & Retraining
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                             height_shift_range=0.1, zoom_range=0.1)
datagen.fit(x_train)

aug_history = cnn_model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=64),
    validation_data=(x_test, y_test_cat),
    epochs=5
)

# Save augmented model curves
plt.plot(aug_history.history['accuracy'], label='Train Acc')
plt.plot(aug_history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Augmented CNN Training Accuracy')
plt.savefig('aug_cnn_accuracy.png')
plt.close()

# 4. Model Evaluation
y_pred = np.argmax(cnn_model.predict(x_test), axis=1)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 5. Transfer Learning with MobileNetV2
print("Running Transfer Learning with MobileNetV2...")
mobilenet = MobileNetV2(input_shape=(96,96,3), include_top=False, weights='imagenet', pooling='avg')
mobilenet.trainable = False

# Prepare dataset for MobileNet (resize + 3 channels)
x_train_rgb = np.repeat(tf.image.resize(x_train, (96,96)), 3, axis=-1)
x_test_rgb = np.repeat(tf.image.resize(x_test, (96,96)), 3, axis=-1)

transfer_model = models.Sequential([
    mobilenet,
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
transfer_model.fit(x_train_rgb, y_train_cat, validation_data=(x_test_rgb, y_test_cat), epochs=3)

# Save model
cnn_model.save('cnn_model.h5')
transfer_model.save('mobilenet_model.h5')
print("Models saved successfully.")

