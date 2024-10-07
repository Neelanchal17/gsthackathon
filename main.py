
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


# Testing all the differnt models that we can use
# Hash Table of all Models that we can use for this problem set
models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(), 
    "XGBoost": XGBClassifier()
}

# To make sure both the data are in pandas DataFrame
print(type(train_data_x), type(train_data_y))


# Traversing through each Model and finding out the different metrics

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(train_data_x, train_data_y['target'])

    y_train_preds = model.predict(train_data_x)
    y_test_preds = model.predict(test_data_x)

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

# As we can see from the results that XGBoost performs the best amongst all the models in every metric
model = XGBClassifier()
model.fit(train_data_x, train_data_y['target'])
preds = model.predict(test_data_x)


# Neural Network Implementation
# Tensorflow Neural Network Model (Testing it's accuracy,recall,f1 etc) 
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


# Getting the output to be either 0 or 1 instead of being a probabilty
for i in preds:
    if i > 0.5:
        preds[i] == 1
    elif i < 0.5:
        preds[i] == 0
preds = [1 if p >= 0.5 else 0 for p in preds]

# Calculating all the Metrics for evaluating the Model
acc = accuracy_score(test_data_y, preds)
f1 = f1_score(test_data_y, preds, average='weighted')
precision = precision_score(test_data_y, preds, average='binary')
recall = recall_score(test_data_y,preds)
roc_auc = roc_auc_score(test_data_y, preds)



# Printing all the Metrics
print(f'Accuracy: {acc}')
print(f"F1 Score: {f1}")
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Roc Auc Score: {roc_auc}')
print(classification_report(test_data_y, preds))

# Confusion Matrix for the final testing prediction
import seaborn as sns
cm = confusion_matrix(test_data_y, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()




