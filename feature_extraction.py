import os
import cv2
import numpy as np

from skimage.feature import hog

# Dataset path
train_path = "archive/data/train"

X = []
y = []

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

for label in range(7):

    folder_path = os.path.join(train_path, str(label))

    print(f"Processing {emotions[label]} images...")

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (64, 64))

        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Save features
np.save("X_features.npy", X)
np.save("y_labels.npy", y)

print("\nFeatures Saved Successfully!")
print("X Shape:", X.shape)
print("y Shape:", y.shape)