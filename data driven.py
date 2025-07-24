# A Data-Driven Approach for Customer Lifetime Value (CLV) in E-Commerce

# Step 1: Importing Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns

""" Snippet Generated Select with current selected text """
""" End of the Snippet Generated Select """


import warnings
import os
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

# Step 2: Load Dataset
df = pd.read_csv('Datasets/Dataset.csv')

# Step 3: Visual Exploration
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()

# Count plot for Target classes
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='Target', data=df)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.title('Count Plot for Target Variable')
plt.show()

# Step 4: Preprocessing - Label Encoding
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Step 5: Splitting Features and Target
target_column = 'Target'
X = df.drop(columns=[target_column])
y = df[target_column]
labels = df['Target'].unique()

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

# Step 7: Metric Calculation Function
precision = []
recall = []
fscore = []
accuracy = []

def calculateMetrics(algorithm, testY, predict):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    print(f"{algorithm} Accuracy    : {a}")
    print(f"{algorithm} Precision   : {p}")
    print(f"{algorithm} Recall      : {r}")
    print(f"{algorithm} FSCORE      : {f}")

    print(f"\n{algorithm} classification report\n")
    print(classification_report(predict, testY, target_names=[str(l) for l in labels]))

    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="Blues", fmt="g")
    ax.set_ylim([0, len(labels)])
    plt.title(f"{algorithm} Confusion Matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

# Step 8: Lasso Classifier
if os.path.exists('model/Lasso.pkl'):
    LC = joblib.load('model/Lasso.pkl')
    print("Lasso model loaded successfully.")
else:
    LC = Lasso()
    LC.fit(X_train, y_train)
    os.makedirs('model', exist_ok=True)
    joblib.dump(LC, 'model/Lasso.pkl')
    print("Lasso model trained and saved successfully.")

predict_lasso = LC.predict(X_test).round().astype(int)
calculateMetrics("Lasso Classifier", y_test, predict_lasso)

# Step 9: Decision Tree Classifier
if os.path.exists('model/DTC.pkl'):
    DTC = joblib.load('model/DTC.pkl')
    print("Decision Tree model loaded successfully.")
else:
    DTC = DecisionTreeClassifier()
    DTC.fit(X_train, y_train)
    joblib.dump(DTC, 'model/DTC.pkl')
    print("Decision Tree model trained and saved successfully.")

predict_dtc = DTC.predict(X_test)
calculateMetrics("Decision Tree Classifier", y_test, predict_dtc)

# Step 10: Predicting on New Test Data
test = pd.read_csv('Datasets/testdata.csv')

# Label encoding test data
for col in test.columns:
    if test[col].dtype == 'object':
        test[col] = le.fit_transform(test[col])

# Predict using DTC
predictions = DTC.predict(test)

# Add predictions to test DataFrame
test['Prediction'] = [labels[p] for p in predictions]

# Display final test predictions
print("\nTest Data Predictions:")
print(test)
