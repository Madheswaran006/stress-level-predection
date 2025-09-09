import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, Response, jsonify
import cv2, random, os

app = Flask(__name__, template_folder='templates')

# ----------------------------
# Load dataset & train model
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
# Camera setup (Render-safe)
# ----------------------------
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    camera = None

def generate_frames():
    while True:
        if camera is None:
            # No camera available (Render environment)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            success, frame = camera.read()
            if not success:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/camera_feed')
def camera_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------------------
# Stress Analysis
# ----------------------------
@app.route('/analyze', methods=['GET'])
def analyze():
    # For now: simulate result
    result = random.choice(["Low Stress", "Moderate Stress", "High Stress"])
    return jsonify({"stress_level": result})

# ----------------------------
# Main page
# ----------------------------
@app.route('/')
def home():
    return render_template('login.html')  # or index.html

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

