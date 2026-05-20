from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


DATASET_ROOT = Path("dataset/C2A_Dataset/C2A_Dataset/new_dataset3")
MODEL_PATH = Path("models/disaster_image_model.pkl")
IMAGE_SIZE = (64, 64)
MAX_IMAGES_PER_CLASS = 400

LABELS = {
    "collapsed_building": "Collapsed Building",
    "fire": "Fire",
    "flood": "Flood",
    "traffic_incident": "Traffic Incident",
}


def label_from_filename(filename):
    for prefix, label in LABELS.items():
        if filename.startswith(prefix):
            return label
    return None


def image_features(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.resize(img, IMAGE_SIZE)
    return (img.astype(np.float32) / 255.0).flatten()


def load_split(split):
    images_dir = DATASET_ROOT / split / "images"
    data = []
    labels = []
    counts = {label: 0 for label in LABELS.values()}

    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label = label_from_filename(image_path.name)
        if label is None:
            continue
        if counts[label] >= MAX_IMAGES_PER_CLASS:
            continue

        data.append(image_features(image_path))
        labels.append(label)
        counts[label] += 1

    return np.array(data), np.array(labels)


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")

model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)

MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(
    {
        "model": model,
        "image_size": IMAGE_SIZE,
        "labels": LABELS,
    },
    MODEL_PATH,
)

print("Model Trained Successfully")
print(f"Validation Accuracy: {accuracy:.2%}")
print(f"Saved model: {MODEL_PATH}")
