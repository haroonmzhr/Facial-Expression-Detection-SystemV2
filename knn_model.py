import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load saved features
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

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

# KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)

print("\nTraining KNN Model...")

# Train model
knn_model.fit(X_train, y_train)

print("KNN Training Completed!")

# Predictions
knn_pred = knn_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, knn_pred)

print("\nKNN Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, knn_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, knn_pred)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Plot
plt.figure(figsize=(10,8))

plt.imshow(cm, cmap='Oranges')

plt.title("KNN Confusion Matrix", fontsize=16)

plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)

plt.xticks(np.arange(len(emotion_labels)), emotion_labels, rotation=45)
plt.yticks(np.arange(len(emotion_labels)), emotion_labels)

# Add numbers inside boxes
for i in range(len(emotion_labels)):
    for j in range(len(emotion_labels)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="black"
        )

plt.colorbar()

plt.tight_layout()

plt.show()

#import joblib

#joblib.dump(knn_model, "knn_emotion_model.pkl")

#print("\nKNN Model Saved!")