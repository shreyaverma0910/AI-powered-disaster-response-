from flask import Flask, render_template, request, url_for
import cv2
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

model_bundle = joblib.load("models/disaster_image_model.pkl")
model = model_bundle["model"]
IMAGE_SIZE = tuple(model_bundle["image_size"])

UPLOAD_FOLDER = "static/uploads"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    filename = secure_filename(file.filename)

    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    file.save(path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return render_template(
            'index.html',
            prediction="Could not read uploaded image"
        )

    img = cv2.resize(img, IMAGE_SIZE)

    img = (img.astype(np.float32) / 255.0).flatten().reshape(1, -1)

    prediction = model.predict(img)

    return render_template(
        'index.html',
        prediction=prediction[0],
        image_path=url_for('static', filename=f'uploads/{filename}')
    )


if __name__ == "__main__":
    app.run(debug=True)
