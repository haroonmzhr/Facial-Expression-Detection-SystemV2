import cv2
import joblib
import numpy as np

from skimage.feature import hog

# Load trained model
model = joblib.load("rf_emotion_model.pkl")

# Emotion labels
emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Open webcam
cap = cv2.VideoCapture(0)

print("Press Q to Quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # Extract face
        face = gray[y:y+h, x:x+w]

        # Resize
        face = cv2.resize(face, (64, 64))

        # HOG features
        features = hog(
            face,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )

        # Reshape
        features = features.reshape(1, -1)

        # Prediction
        prediction = model.predict(features)

        emotion = emotion_labels[prediction[0]]

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Show emotion text
        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # Display frame
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Quit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()