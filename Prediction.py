import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pickle

# Fetch dataset
heart_data = fetch_ucirepo(id=45)

# Extract features and target
X = heart_data.data.features
y = heart_data.data.targets

# Get column names from metadata
column_names = getattr(heart_data.variables, 'feature_names', None)

# Convert data to pandas DataFrame
heart_df = pd.DataFrame(data=X, columns=column_names)

# Add the target column to the DataFrame
heart_df['target'] = y

# Map target labels to binary classification (0 for absence, 1 for presence of heart disease)
heart_df['target'] = heart_df['target'].replace({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

# Separate features (X) and target (y)
X = heart_df.drop(columns=['target'])
y = heart_df['target']

# Initialize Stratified K-Fold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
accuracies = []
for train_index, test_index in stratified_kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Impute missing values in the training and testing data
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Initialize KMeans with optimized parameters
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=50, max_iter=500, random_state=42)

    # Fit KMeans on the scaled training data
    kmeans.fit(X_train_scaled)

    # Predict cluster labels for the testing data
    test_cluster_labels = kmeans.predict(X_test_scaled)

    # Assign labels based on majority class in each cluster
    cluster_0_label = np.bincount(y_test[test_cluster_labels == 0]).argmax()
    cluster_1_label = np.bincount(y_test[test_cluster_labels == 1]).argmax()
    predicted_labels = np.where(test_cluster_labels == 0, cluster_0_label, cluster_1_label)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_labels)
    accuracies.append(accuracy)

# Average accuracy across folds
avg_accuracy = np.mean(accuracies)
print("Average Cross-Validation Accuracy with KMeans:", avg_accuracy)

# Save the trained KMeans model
model_filename = 'heart-disease-prediction-kmeans-model.pkl'
pickle.dump(kmeans, open(model_filename, 'wb'))
