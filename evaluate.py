import argparse, os
from data_prep import get_mnist
from utils.plots import plot_confusion
import tensorflow as tf
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Path to model .h5 file')
args = parser.parse_args()

os.makedirs('artifacts', exist_ok=True)

# Data
(_, _), (x_test, y_test), _, _ = get_mnist()

# Load model & predict
model = tf.keras.models.load_model(args.model)
y_prob = model.predict(x_test, verbose=0)
y_pred = y_prob.argmax(axis=1)

# Metrics
report = classification_report(y_test, y_pred, digits=4)
with open('artifacts/classification_report.txt', 'w') as f:
    f.write(report)
print(report)

# Confusion matrix
plot_confusion(y_test, y_pred, 'artifacts/confusion_matrix.png', class_names=[str(i) for i in range(10)])
print('Saved confusion matrix to artifacts/confusion_matrix.png')

