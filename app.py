import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, Response, jsonify
import cv2, random

app = Flask(__name__, template_folder='templates')

# ----------------------------
# Load dataset & train model (still here if you want to use form later)
# ----------------------------
data = pd.read_csv("StressLevelDataset.csv")
encoder = LabelEncoder()
data["stress_level"] = encoder.fit_transform(data["stress_level"])

X = data.drop("stress_level", axis=1)
y = data["stress_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree_clf = DecisionTreeClassifier(max_depth=7, random_state=100)
tree_clf.fit(X_train, y_train)

# ----------------------------
# Camera setup
# ----------------------------
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/camera_feed')
def camera_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------------------
# Simulated Stress Analysis
# ----------------------------
@app.route('/analyze', methods=['GET'])
def analyze():
    ret, frame = camera.read()
    if not ret:
        return jsonify({"stress_level": "Camera Error"})

    # 👉 Simulate stress detection (random result for demo)
    result = random.choice(["Low Stress", "Medium Stress", "High Stress"])
    return jsonify({"stress_level": result})

# ----------------------------
# Login route now shows camera UI
# ----------------------------
@app.route('/')
def login():
    return render_template('login.html')

# ----------------------------
# Run app
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
