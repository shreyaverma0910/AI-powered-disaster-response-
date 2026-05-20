import joblib
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained scikit-learn model (.pkl)
model = joblib.load('models/disaster_lstm_model.h5')

# Prediction classes
classes = [
    'Earthquake Damage',
    'Flood Disaster',
    'Safe Area',
    'Survivor Detected'
]

# Image path
img_path = 'test.jpg'

# Load image
img = image.load_img(img_path, target_size=(128,128))

# Convert image to array
img_array = image.img_to_array(img)

# Flatten image for scikit-learn model (expects 1D features, not 3D tensor)
img_array = img_array.flatten()

# Expand dimensions to match (1, n_features)
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

# Get result
result = classes[np.argmax(prediction)]
confidence = round(np.max(prediction) * 100, 2)

# Display output
print("Prediction:", result)
print("Confidence:", confidence, "%")
