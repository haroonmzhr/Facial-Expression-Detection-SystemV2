import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

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

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA
pca = PCA(n_components=300)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("\nPCA Applied Successfully!")
print("New Shape:", X_train.shape)

# SVM
svm_model = LinearSVC(C=1.5,max_iter=5000)

print("\nTraining SVM Model...")

svm_model.fit(X_train, y_train)

print("SVM Training Completed!")

# Prediction
y_pred = svm_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nSVM Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Plot
plt.figure(figsize=(10,8))

plt.imshow(cm, cmap='Blues')

plt.title("SVM Confusion Matrix", fontsize=16)

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
#joblib.dump(svm_model, "svm_emotion_model.pkl")

# Save PCA
#joblib.dump(pca, "pca_model.pkl")

#print("\nModel and PCA Saved!")