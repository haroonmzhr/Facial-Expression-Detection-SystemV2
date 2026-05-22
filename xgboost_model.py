import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

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

# Create XGBoost model
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='mlogloss'
)

print("\nTraining XGBoost Model...")

# Train model
xgb_model.fit(X_train, y_train)

print("XGBoost Training Completed!")

# Predictions
xgb_pred = xgb_model.predict(X_test)

# Accuracy
xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("\nXGBoost Accuracy:", xgb_accuracy)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, xgb_pred))

# Save model
#joblib.dump(xgb_model, "xgboost_model.pkl")

#print("\nXGBoost Model Saved!")

# Confusion Matrix
cm = confusion_matrix(y_test, xgb_pred)

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

plt.imshow(cm, cmap='Reds')

plt.title("XGBoost Confusion Matrix", fontsize=16)

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