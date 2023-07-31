import numpy as np
import random
import os
import pickle
import math
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import backend as K
from tensorboard import notebook
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import scipy.stats as stats
import pandas as pd
from sklearn.metrics import mean_squared_error
import re
from scipy.stats import rankdata
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'column_name' is the column you want to convert

from scipy.stats import spearmanr
from sklearn.metrics import make_scorer

def spearman_correlation(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = y_true.shape[0]
    spearman_corr = []
    for i in range(y_true.shape[1]):
        rho, _ = spearmanr(y_true[:, i], y_pred[:, i])
        spearman_corr.append(rho)
    avg_spearman_corr = np.mean(spearman_corr)
    return avg_spearman_corr

def tanh_tilde(x):
    return 1.5 * (K.tanh(x) + 1) + 1

def remove_complex_parts(value):
    value = str(value)  # Convert to string if not already
    value = value.replace('(', '').replace(')', '')  # Remove brackets
    value = value.replace('-0j', '').replace('+0j', '')  # Remove '-0j' and '+0j'
    return float(value)  # Convert to float

df = pd.read_csv(r"C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\TrainComplete.csv", sep=';')

x = df.iloc[:, 4:]
y = df.iloc[:, 0:4]
num_features = df.shape[1] - 4
print(df.isnull().any())

# Check for missing values in the "RECT" column
missing_values = df['Rect'].isnull().sum()
print("Number of missing values in RECT:", missing_values)

# Check the data type of the "RECT" column
data_type = df['Rect'].dtype
print("Data type of RECT:", data_type)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=70, random_state=42)

NN_train = y_train.iloc[:, 0]
GR_train = y_train.iloc[:, 1]
NI_train = y_train.iloc[:, 2]
FI_train = y_train.iloc[:, 3]


# Standardize the input features
scaler = StandardScaler()

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

x = scaler.fit_transform(x)

# Set the number of folds (K)
k_folds = 5

# Initialize the K-fold cross-validator
kf = KFold(n_splits=k_folds, shuffle=True)

param_grid = [9,29,49,69,89,109,129,149,169,189,209]

for ii in range(len(param_grid)):
    neigbors = param_grid[ii]

    knn1 = KNeighborsRegressor(n_neighbors=neigbors)
    knn2 = KNeighborsRegressor(n_neighbors=neigbors)
    knn3 = KNeighborsRegressor(n_neighbors=neigbors)
    knn4 = KNeighborsRegressor(n_neighbors=neigbors)

    spearman_folds = []
    k = 0
    for train_index, test_index in kf.split(x):
        k +=1
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


        NN_train = y_train.iloc[:, 0]
        GR_train = y_train.iloc[:, 1]
        NI_train = y_train.iloc[:, 2]
        FI_train = y_train.iloc[:, 3]

        # Fit the model to the training data
        knn1.fit(x_train, NN_train)
        knn2.fit(x_train, GR_train)
        knn3.fit(x_train, NI_train)
        knn4.fit(x_train, FI_train)
        NNpred = knn1.predict(x_test)
        GRpred = knn2.predict(x_test)
        NIpred = knn3.predict(x_test)
        FIpred = knn4.predict(x_test)
        y_true = np.array(y_test)

        # Combine the predicted values of all models into a single array
        predicted_values = np.column_stack((NNpred, GRpred, NIpred, FIpred))

        # Obtain the ranks of the values
        predicted_ranks = np.apply_along_axis(rankdata, 1, predicted_values)

        # Convert ranks to integers
        predicted_ranks = predicted_ranks.astype(int)

        # Calculate Spearman's correlation coefficient
        rhos = []
        for i in range(len(y_true)):
            rho, _ = spearmanr(predicted_ranks[i], y_true[i])
            rhos.append(rho)
        mean_spearman_fold = np.mean(rhos)
        spearman_folds.append(mean_spearman_fold)
        print("fold completed: {}/5".format(k))

  
    mean_spearman = np.mean(spearman_folds)
    

    result = [neigbors, mean_spearman]
    with open(r'C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\KFOLDKNN', 'a', newline='') as csvfile1:
        writer1 = csv.writer(csvfile1)
        writer1.writerow(result)





y_test_list = y_test.values.tolist()
for i in range(len(NNpred)):
    predicted_values = [NNpred[i], GRpred[i], NIpred[i], FIpred[i]]
    true_values = y_test_list[i]
    print("Predicted:", [f"{value:.3f}" for value in predicted_values], "True:", true_values)


