import cv2
import joblib
import numpy as np

from skimage.feature import hog

# Load trained KNN model
model = joblib.load("knn_emotion_model.pkl")

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

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Process each face
    for (x, y, w, h) in faces:

        # Extract face region
        face = gray[y:y+h, x:x+w]

        # Resize image
        face = cv2.resize(face, (64, 64))

        # Extract HOG features
        features = hog(
            face,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )

        # Reshape for prediction
        features = features.reshape(1, -1)

        # Predict emotion
        prediction = model.predict(features)

        emotion = emotion_labels[prediction[0]]

        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 165, 255),
            2
        )

        # Display emotion text
        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 165, 255),
            2
        )

    # Show webcam
    cv2.imshow("Real-Time KNN Emotion Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
