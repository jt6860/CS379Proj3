# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:08:58 2024

@author: John Torres
Course: CS379 - Machine Learning
Project 3: German Credit Fraud Dataset - Class Prediction using Random Forest
Supervised Algorithm: Random Forest
"""

from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Helper function for column decoding.
def decode_columns(dataframe, columns):
    for column in columns:
        if dataframe[column].dtype == object:
            dataframe[column] = dataframe[column].str.decode('utf-8')

# Load Credit Fraud data from .arff file.
data, meta = arff.loadarff('credit_fraud.arff')

# Create dataframe from ingested data.
dataframe = pd.DataFrame(data)

# Decode any columns where dtype is object.
columns_to_decode = dataframe.columns
decode_columns(dataframe, columns_to_decode)

# Check for Missing Values.  Yields 0 when used with 'credit_fraud.arff'
print("Null Counts")
print(dataframe.isnull().sum())

# Split dataframe into X and y sets.
X = dataframe.drop('class', axis=1)
y = dataframe["class"]

# Encode the target variable as discrete classes.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Change categorical values to indicator values.
X  = pd.get_dummies(X)
X_frame = pd.DataFrame(X)

# Scale the data.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split X and y into X_train, X_test, y_train, y_test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Instantiate and fit the Random Forest Classifier for prediction.
# Gridsearch identified best estimators:   RandomForestClassifier(max_depth=6, max_features=None, max_leaf_nodes=9)
# RandomSearch identified best estimators: RandomForestClassifier(max_depth=6, max_features=None, max_leaf_nodes=9, n_estimators=25)
classifier = RandomForestClassifier(max_depth=6, max_features=None, max_leaf_nodes=9, n_estimators=25)
classifier = classifier.fit(X_train, y_train)
# 74 - 76% accuracy with scaler and gridsearchcv best estimators
# 74 - 76% accuracy with scaler and randomsearchcv best estimators

# Make predictions using the trained classifier.
predicted = classifier.predict(X_test)
confusionMatrix = confusion_matrix(y_test, predicted)
classes = classifier.classes_
        
# Generate and visualize Feature Importance Scores
feature_scores = pd.Series(classifier.feature_importances_, index=X_frame.columns).sort_values(ascending=False)
f, ax = plt.subplots(figsize=(20, 14))
ax = sns.barplot(x=feature_scores, y=feature_scores.index, data=dataframe)
ax.set_title("Feature Importance Scores")
ax.set_yticklabels(feature_scores.index)
ax.set_xlabel("Feature Importance Score")
ax.set_ylabel("Features")
plt.show()

# Print Accuracy Score and Classification Report, plot Confusion Matrix heatmap.
print('Confusion Matrix:')
print(confusion_matrix(y_test, predicted))
print('Accuracy Score:', accuracy_score(y_test, predicted))
print('Report:')
print(classification_report(y_test, predicted, zero_division=0))

# Create and show Confusion Matrix display.    
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classes)
cm_display.plot(include_values=True, xticks_rotation='horizontal')
plt.show()
