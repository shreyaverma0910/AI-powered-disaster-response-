import joblib
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained scikit-learn model (.pkl)
model = joblib.load('models/disaster_image_model.pkl')

# Prediction classes
classes = [
    'Collapsed Building',
    'Fire',
    'Flood',
    'Traffic Incident'
]

# Image path
img_path = 'test.jpg'

# Load image
img = image.load_img(img_path, target_size=(128, 128))

# Convert image to array
img_array = image.img_to_array(img)

# Flatten image for scikit-learn model
img_array = img_array.flatten()

# Expand dimensions
img_array = np.expand_dims(img_array, axis=0)

# Prediction probabilities
prediction = model.predict_proba(img_array)

# Highest probability
max_prob = np.max(prediction)

# Predicted class index
predicted_index = np.argmax(prediction)

# Confidence percentage
confidence = round(max_prob * 100, 2)

# Threshold for disaster detection
threshold = 70

# Final result
if confidence < threshold:
    result = "Normal Image"
else:
    result = classes[predicted_index]

# Display output
print("Prediction:", result)
print("Confidence:", confidence, "%")