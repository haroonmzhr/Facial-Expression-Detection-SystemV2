import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# Random Forest Model
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
accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Plot
plt.figure(figsize=(10,8))

plt.imshow(cm, cmap='Greens')

plt.title("Random Forest Confusion Matrix", fontsize=16)

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

# Save model
#joblib.dump(rf_model, "rf_emotion_model.pkl")

#print("\nModel Saved Successfully!")