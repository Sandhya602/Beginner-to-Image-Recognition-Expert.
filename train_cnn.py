import os
from data_prep import get_mnist, get_augmented_flow
from model_cnn import build_cnn
from utils.plots import plot_learning_curves

os.makedirs('artifacts', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Load MNIST
(x_train, y_train), (x_test, y_test), input_shape, num_classes = get_mnist()

# Build & train baseline CNN
model = build_cnn(input_shape=input_shape, num_classes=num_classes)
history = model.fit(
    x_train, y_train, 
    validation_data=(x_test, y_test), 
    epochs=5, batch_size=64, verbose=1
)

plot_learning_curves(history, 'artifacts/cnn_accuracy.png')
model.save('checkpoints/cnn_model.h5')
print('Baseline CNN training complete. Saved to checkpoints/cnn_model.h5')

# Augmentation pass (improves generalization)
flow = get_augmented_flow(x_train, y_train, batch_size=64)
aug_history = model.fit(
    flow,
    validation_data=(x_test, y_test),
    epochs=5, verbose=1
)
plot_learning_curves(aug_history, 'artifacts/aug_cnn_accuracy.png')
model.save('checkpoints/cnn_model_aug.h5')
print('Augmented CNN training complete. Saved to checkpoints/cnn_model_aug.h5')

