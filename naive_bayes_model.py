import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
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

# Create Naive Bayes model
nb_model = GaussianNB()

print("\nTraining Naive Bayes Model...")

# Train model
nb_model.fit(X_train, y_train)

print("Naive Bayes Training Completed!")

# Predictions
nb_pred = nb_model.predict(X_test)

# Accuracy
nb_accuracy = accuracy_score(y_test, nb_pred)

print("\nNaive Bayes Accuracy:", nb_accuracy)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, nb_pred))

# Save model
#joblib.dump(nb_model, "naive_bayes_model.pkl")

#print("\nNaive Bayes Model Saved!")

# Confusion Matrix
cm = confusion_matrix(y_test, nb_pred)

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

# Visualization
plt.figure(figsize=(10,8))

plt.imshow(cm, cmap='Purples')

plt.title("Naive Bayes Confusion Matrix", fontsize=16)

plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)

plt.xticks(np.arange(len(emotion_labels)), emotion_labels, rotation=45)
plt.yticks(np.arange(len(emotion_labels)), emotion_labels)

# Add numbers
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