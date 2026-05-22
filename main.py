import os
import cv2
import numpy as np

from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

# Dataset path
train_path = "archive/data/train"

# Empty lists
X = []
y = []

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Loop through each emotion folder
for label in range(7):

    folder_path = os.path.join(train_path, str(label))

    print(f"Processing {emotions[label]} images...")

    # Loop through images
    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Skip if image not loaded
        if img is None:
            continue

        # Resize image
        img = cv2.resize(img, (64, 64))

        # Extract HOG features
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )

        # Store features and label
        X.append(features)
        y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)
from sklearn.model_selection import train_test_split

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed!")

# Print shapes
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)

# Print dataset information
print("\nDataset Loaded Successfully!")
print("Feature Matrix Shape:", X.shape)
print("Labels Shape:", y.shape)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Create SVM model
svm_model = SVC(kernel='rbf')

print("\nTraining SVM Model...")

# Train model
svm_model.fit(X_train, y_train)

print("SVM Training Completed!")

# Predictions
y_pred = svm_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nSVM Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nTraining Random Forest Model...")

# Train model
rf_model.fit(X_train, y_train)

print("Random Forest Training Completed!")

# Predictions
rf_pred = rf_model.predict(X_test)

# Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

# Classification Report
print("\nRandom Forest Classification Report:\n")
print(classification_report(y_test, rf_pred))