'''
Here we'll do all the pre-processing and the processing steps on the data and further calibrate it to suite our needs
Categorical Columns - 0, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21
Numeric COlumns - 1,2,3,4,5,6,7,8,9,14,15

'''


# Imports 
import numpy as np
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


print(np.__version__)
print(pd.__version__)
'''
Version of different libraries and language used:
Python version - Python 3.12.6
Numpy version - 1.26.4
Pandas version - 2.2.3
'''

# Getting the data
train_data_x = pd.read_csv(r'/Train_60/Train_60/X_Train_Data_Input.csv')
train_data_y = pd.read_csv(r'/Train_60/Train_60/Y_Train_Data_Target.csv')

test_data_x = pd.read_csv(r'/Test_20/Test_20/X_Test_Data_Input.csv')
test_data_y = pd.read_csv(r'/Test_20/Test_20/Y_Test_Data_Target.csv')


# Getting all the column names
print(train_data_x.columns)
print(train_data_y.columns)


# Droping the ID column for processing the data
train_data_x.drop(columns=['ID'], inplace=True)
test_data_y.drop(columns=['ID'], inplace=True)
train_data_y.drop(columns=['ID'], inplace=True)
test_data_x.drop(columns=['ID'], inplace=True)
print('ID Column removed...')



# No of missing values in each column:
print(train_data_x.isnull().sum())
print(test_data_x.isnull().sum())



# Feature Selection 
# From the results, we can see the Column 9 has the largest no of missing values
# So we'll remove that column
train_data_x.drop(columns=["Column9"], inplace=True)
test_data_x.drop(columns=["Column9"], inplace=True)
print('Column 9 removed...')
# We could drop Column 14, but the output doesn't improve our metrics
# train_data_x.drop(columns=["Column14"], inplace=True)
# test_data_x.drop(columns=["Column14"], inplace=True)




print(train_data_x.isnull().sum())
print(test_data_x.isnull().sum()) 



# I implemented KNN Imputer before and it was very computational heavy, so I'll stick with Simple Imputation
# Now We'll use Simple Imputer to get data values that are NA to a value that we can evaluate
# For Binary Data

binary_cols = [
    'Column10', 'Column11',
       'Column12', 'Column13','Column14','Column19', 'Column20', 'Column21'
]

imputer_binary = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_data_x[binary_cols] = imputer_binary.fit_transform(train_data_x[binary_cols])


test_data_x[binary_cols] = imputer_binary.fit_transform(test_data_x[binary_cols])

# # For Numeric Data
# # Numeric COlumns - 1,2,3,4,5,6,7,8,9,14,15
numeric_cols = [
    'Column1', 'Column2', 'Column3', 'Column4', 'Column5',
       'Column6', 'Column7', 'Column8', 'Column15'
] 

imputer_numeric = SimpleImputer(missing_values=np.nan, strategy="median")

train_data_x[numeric_cols] = imputer_numeric.fit_transform(train_data_x[numeric_cols])


# # Using the fitted imputer to transform the test data
test_data_x[numeric_cols] = imputer_numeric.transform(test_data_x[numeric_cols])

# # Applying the same scaling to both train and test sets
scaler = StandardScaler()  
train_data_x[numeric_cols] = scaler.fit_transform(train_data_x[numeric_cols])
test_data_x[numeric_cols] = scaler.transform(test_data_x[numeric_cols])

imputer_column0 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# Fiting the imputer on training data
train_data_x['Column0'] = imputer_column0.fit_transform(train_data_x[['Column0']])

# # # Applying the imputation to the test data
test_data_x['Column0'] = imputer_column0.transform(test_data_x[['Column0']])

test_data= test_data_x.dropna(subset=['Column0'])
test_data = test_data.loc[test_data_x.index] 
   
# Evaluating the binary columns in the dataset
# 10, 11, 12, 13, 19,20, 21 - columns that are binary
column_dic = {
    10: 'Column10',
    11: 'Column11',
    12: 'Column12',
    13: 'Column13',
    19: 'Column19',
    20: 'Column20',
    21: 'Column21',
}
for j in list(column_dic.values()):
    data_values = train_data_x[j]
    count_0 = 0
    count_1 = 0
    for i in data_values:
        if i == 0:
            count_0 += 1
        elif i == 1:
            count_1 += 1

    print(f'Count of 0 in {j}: {count_0}')
    print(f'Count of 1 in {j}: {count_1}')
    print('\n')


# To reduce Class imbalance 
# When implemented this decreases the precision and increases the recall
# If we implement SMOTE, we face this conundrum of choosing either the precision or the recall as our main priority
# smote = SMOTE(sampling_strategy='auto', random_state=42)
# train_data_x, train_data_y = smote.fit_resample(train_data_x, train_data_y)


print(train_data_x.isnull().sum())
print(test_data_x.isnull().sum()) 

train_data_x.boxplot()
plt.xticks(rotation=90)
plt.title("Boxplot to Visualize Outliers")
plt.show()