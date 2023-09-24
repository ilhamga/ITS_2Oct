from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import argparse

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Input image.')
parser.add_argument('--input', type=str, help='Path to the input image')
parser.add_argument('--model', default="material/model/keras_model.h5", type=str, help='Path to the model')
parser.add_argument('--label', default="material/model/labels.txt", type=str, help='Path to the label file')
args = parser.parse_args()

# Load the model
model = load_model(args.model, compile=False)

# Load the labels
class_names = open(args.label, "r").readlines()

# Load the image
image = cv2.imread(args.input)

# Resize the raw image into (224-height,224-width) pixels
image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

# Make the image a numpy array and reshape it to the models input shape.
image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

# Normalize the image array
image = (image / 127.5) - 1

# Predicts the model
prediction = model.predict(image)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
