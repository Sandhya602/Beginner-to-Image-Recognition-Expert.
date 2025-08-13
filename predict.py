import argparse, cv2, numpy as np, tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str)
parser.add_argument('--image', required=True, type=str, help='Path to single image for prediction')
args = parser.parse_args()

model = tf.keras.models.load_model(args.model)

img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f'Could not read image: {args.image}')

img = cv2.resize(img, (28,28))
img = img.astype('float32')/255.0
img = np.expand_dims(img, axis=(0,-1))  # (1,28,28,1)

pred = model.predict(img, verbose=0)[0]
label = int(np.argmax(pred))
conf = float(np.max(pred))

print(f'Predicted: {label} (confidence: {conf:.4f})')

