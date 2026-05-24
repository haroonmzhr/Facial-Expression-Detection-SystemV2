FACIAL EXPRESSION RECOGNITION
USING MACHINE LEARNING
 
Course: Machine Learning Lab
Date: May 19, 2026
Members: Muhammad Ahmad, Muhammad Haroon Mazhar
Instructor: Dr. Farman Ali
Table of Contents
	Introduction
	Objectives
	Dataset Description
	Data Preprocessing
	Feature Extraction using HOG
	Machine Learning Models
	Model Training and Testing
	Performance Evaluation
	Performance Comparison Table
	Accuracy Comparison Graph
	Real-Time Emotion Detection
	Challenges and Limitations
	Results and Discussion
	Conclusion
	Future Improvements
	References
Introduction
Facial Expression Recognition (FER) is a crucial area of research in computer vision and artificial intelligence, enabling machines to interpret human emotional states. Applications of FER span human-computer interaction (HCI), driver drowsiness detection, mental health monitoring, and smart entertainment systems. This project implements a complete machine learning pipeline to classify human facial expressions into distinct emotional categories using traditional machine learning techniques combined with Histogram of Oriented Gradients (HOG) feature extraction.
Objectives
	Implement a complete image classification workflow.
	Extract structural features from facial images using HOG.
	Train and evaluate LinearSVC, Random Forest, KNN, Naive Bayes, and XGBoost.
	Compare models using Accuracy, Precision, Recall, and F1-Score.
	Assess feasibility for real-time deployment.
Dataset Description
The project utilizes the FER2013 (Facial Expression Recognition 2013) dataset, specifically the structured compilation curated by Uladzislau Astrashab in ImageFolder format.

	Volume and Class Distributions

The total dataset contains 35,887 grayscale facial images fixed at an individual resolution of 48×48 pixels. The samples are categorically separated into 7 primary emotional classes:
	Angry 2. Disgust (Highly underrepresented baseline category)
	Fear
	Happy (Most heavily represented baseline category)
	Sad
	Surprise
	Neutral

	Project Utilization Split

To perform rigorous machine learning validation, an 80/20 stratified split configuration was utilized across the categorical subfolders:
	Total Images Utilized: 35,887 images.
	Training Subset Size (80%): 28,709 images were used to optimize tree boundaries, establish cluster points, and fit hyperplane vectors.
	Testing Subset Size (20%): 7,178 images were fully withheld during training to serve as an unbiased benchmark for compiling the final evaluation matrices.
The dataset introduces robust real-world variables, such as minor facial occlusions (glasses, hair, varying hands), inconsistent head-tilt angles, lighting variations, and structural diversity across ages and ethnicities.
Data Preprocessing
The following preprocessing steps were applied:
	Grayscale standardization
	Image resizing to 48×48 pixels
	Pixel normalization to values between 0 and 1
	Data consistency validation for all image samples
Feature Extraction using HOG
Instead of parsing raw pixel arrays directly into the machine learning models—which triggers the curse of dimensionality and fails to evaluate facial topology—the Histogram of Oriented Gradients (HOG) descriptor is applied. HOG shifts focus from raw pixel brightness values to the direction and magnitude of edges across the face.

	Local Mathematical Extraction Pipeline

	Gradient Computation: The pipeline computes first-order horizontal (G_x) and vertical (G_y) edge gradients for every pixel to isolate sharp changes in intensity using localized Sobel operators. The gradient magnitude and orientation angle (θ) are derived via:
∣G∣=√(G_x^2+G_y^2 ),θ=〖tan⁡〗^(-1) (G_y/G_x )

	Spatial Cell Orientation Binning: The 48×48 face grid is split into small, localized cells measuring 8×8 pixels. Within each individual cell, the gradient orientations for all 64 pixels are accumulated into a histogram containing 9 unsigned directional bins ranging from 0^∘ to 180^∘.
	Block Contrast Normalization: Cells are combined into larger, overlapping spatial regions called blocks measuring 2×2 cells (16×16 pixels). The histograms are normalized across each block to minimize variations caused by harsh illumination changes or shadows.

	Resulting Feature Dimensionality

	Cells per Block: 2×2=4 cells.
	Total Blocks across a 48×48 Grid: With an 8×8 pixel stride, the moving window produces 5×5=25 unique overlapping blocks.
	Total Features Extracted per Face: 25 blocks × 4 cells/block × 9 histogram bins = 900 structural numerical features per image.
This process compresses the input data, converting the raw 2,304 pixel variables into a robust, dense 900-dimensional vector that captures structural facial features like eyebrow arches, lip contours, and nose shapes.
are flattened into a highly dense, robust feature vector that uniquely represents the structural profile of the face.
making it memory-efficient and structurally optimal for wide, sparse vectors produced by HOG feature extractions.
Machine Learning Models
The following machine learning models were trained and evaluated:
	Linear Support Vector Classifier (LinearSVC)
	Random Forest
	K-Nearest Neighbors (KNN)
	Naive Bayes
	XGBoost
Model Training and Testing
The dataset was divided into training and testing subsets using standard train-test splitting techniques. Cross-validation techniques were considered to minimize overfitting and improve generalization performance on unseen data.
Performance Evaluation
The models were evaluated using:
	Accuracy
	Precision
	Recall
	F1-Score
	Confusion Matrix Analysis
Performance Comparison Table
Model	Accuracy	Precision	Recall	F1-Score	Training Speed	Real-Time Performance
LinearSVC	43.38%	0.41	0.43	0.42	Slow	Good
Random Forest	46.39%	0.47	0.46	0.44	Medium	Best
KNN	46.04%	0.45	0.46	0.44	Fast	Decent
Naive Bayes	24.15%	0.22	0.24	0.20	Very Fast	Weak
XGBoost	51.80%	0.52	0.51	0.51	Slow	Unstable
Accuracy Comparison Graph
 
Real-Time Emotion Detection
A real-time FER pipeline was implemented using OpenCV. Live webcam frames were processed through face detection, HOG feature extraction, and real-time emotion classification with overlay visualization.
Challenges and Limitations
	Expression overlap and ambiguity
	HOG sensitivity to rotations and side profiles
	Dataset imbalance issues
	Computational overhead for ensemble methods
Results and Discussion
XGBoost achieved the highest accuracy of 51.80%, outperforming other traditional machine learning algorithms. Ensemble methods demonstrated stronger performance due to their ability to capture complex non-linear relationships in HOG feature spaces.
Conclusion
This project successfully developed a complete Facial Expression Recognition pipeline using traditional machine learning approaches and HOG feature extraction. The results demonstrate the effectiveness of ensemble learning for image-based classification tasks.
Future Improvements
	Transition to deep learning architectures such as CNNs
	Data augmentation techniques
	Automated hyperparameter optimization
	GPU acceleration for faster inference
References
	Astrashab U. (2020) Facial expression dataset image folders (fer2013)*. Kaggle Datasets. Available at: https://www.kaggle.com/datasets/astraszab/facial-expression-dataset-image-folders-fer2013
	Goodfellow, I. J., et al. (2013). Challenges in Representation Learning.
	Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection.
	Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
	Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.

