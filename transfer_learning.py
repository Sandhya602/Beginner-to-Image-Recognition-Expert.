import os, tensorflow as tf
from data_prep import get_mnist
from transfer_learning_model import build_mobilenet_head
from utils.plots import plot_learning_curves

os.makedirs('artifacts', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Get MNIST and upscale to 96x96x3 for MobileNetV2
(x_train, y_train), (x_test, y_test), _, num_classes = get_mnist()
x_train = tf.image.resize(x_train, (96,96))
x_test = tf.image.resize(x_test, (96,96))
x_train = tf.repeat(x_train, repeats=3, axis=-1)
x_test = tf.repeat(x_test, repeats=3, axis=-1)

model = build_mobilenet_head(input_shape=(96,96,3), num_classes=num_classes, train_base=False)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64, verbose=1)

plot_learning_curves(history, 'artifacts/transfer_learning_accuracy.png')
model.save('checkpoints/mobilenet_model.h5')
print('Transfer learning complete. Saved to checkpoints/mobilenet_model.h5')

