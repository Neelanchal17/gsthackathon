# Imports
import tensorflow as tf
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from data import test_data_x, train_data_x, test_data_y, train_data_y
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier

# Print the versions of important libraries used
print(tf.__version__)
print(sklearn.__version__)
print(pd.__version__)

'''
Version of different libraries and language used:

Python version - Python 3.12.6
Tensorflow version - 2.17.0
Sklearn version - 1.5.2
Numpy version - 1.26.4
Pandas version - 2.2.3
'''

# Dictionary of all models to be evaluated
models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(), 
    "XGBoost": XGBClassifier()
}

# Confirm that training data is in the form of pandas DataFrames
print(type(train_data_x), type(train_data_y))


for i in range(len(list(models))):
    ''' 
    Iterate over each model and evaluate performance using various metrics
    '''
    model = list(models.values())[i]  # Get the model object
    model.fit(train_data_x, train_data_y['target'])  # Train the model

    y_train_preds = model.predict(train_data_x)  # Predictions on training data
    y_test_preds = model.predict(test_data_x)    # Predictions on testing data

    # Metrics for training data 
    train_acc = accuracy_score(train_data_y, y_train_preds)
    train_f1 = f1_score(train_data_y, y_train_preds, average='weighted')
    train_precision = precision_score(train_data_y, y_train_preds, average='binary')
    train_recall = recall_score(train_data_y, y_train_preds)
    train_roc_curve = roc_auc_score(train_data_y, y_train_preds)

    # Metrics for testing data
    test_acc = accuracy_score(test_data_y, y_test_preds)
    test_f1 = f1_score(test_data_y, y_test_preds, average='weighted')
    test_precision = precision_score(test_data_y, y_test_preds)
    test_recall = recall_score(test_data_y, y_test_preds)
    test_roc_curve = roc_auc_score(test_data_y, y_test_preds)

    # Print model name and evaluation metrics
    print('----------------------------------------------')
    print(list(models.keys())[i])
    print('Training set performance')
    print(f'Accuracy: {train_acc}')
    print(f"F1 Score: {train_f1}")
    print(f'Precision: {train_precision}')
    print(f'Recall: {train_recall}')
    print(f'Roc Auc Score: {train_roc_curve}')

    print('\nTesting set performance\n')
    print(f'Accuracy: {test_acc}')
    print(f"F1 Score: {test_f1}")
    print(f'Precision: {test_precision}')
    print(f'Recall: {test_recall}')
    print(f'Roc Auc Score: {test_roc_curve}')
    print('\n----------------------------------------------\n')

# After comparing, XGBoost is chosen for final testing
model = XGBClassifier()
model.fit(train_data_x, train_data_y['target'])  # Retrain XGBoost on full training data
preds = model.predict(test_data_x)  # Final predictions

# Neural Network Implementation (commented out — placeholder for testing deep learning models)
# The following block builds and trains a neural network model using TensorFlow.
# model0 = tf.keras.models.Sequential(
#     [               
#         tf.keras.layers.Dense(1000, activation='relu'), 
#         tf.keras.layers.Dense(500, activation='relu'), 
#         tf.keras.layers.Dense(100, activation='relu'), 
#         tf.keras.layers.Dense(1,  activation='sigmoid')  
#     ], name = "my_model" 
# )                            

# model0.compile(
#     loss = tf.keras.losses.BinaryCrossentropy(),
#     optimizer = tf.keras.optimizers.Adam(0.001, clipvalue=1.0),
#     metrics=['accuracy',  'precision', 'recall']
# )

# model0.fit(
#     train_data_x, train_data_y,
#     epochs = 100
# )
# preds = model0.predict(test_data_x)

# Convert prediction probabilities to binary output (0 or 1)
# This is a list comprehension version of the loop below it
preds = [1 if p >= 0.5 else 0 for p in preds]

# Metrics to evaluate XGBoost classifier on test data
acc = accuracy_score(test_data_y, preds)
f1 = f1_score(test_data_y, preds, average='weighted')
precision = precision_score(test_data_y, preds, average='binary')
recall = recall_score(test_data_y, preds)
roc_auc = roc_auc_score(test_data_y, preds)

# Print evaluation metrics
print(f'Accuracy: {acc}')
print(f"F1 Score: {f1}")
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Roc Auc Score: {roc_auc}')
print(classification_report(test_data_y, preds))

# Plot confusion matrix using seaborn
import seaborn as sns
cm = confusion_matrix(test_data_y, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


