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

df = pd.read_csv(r"C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\TrainCompleteKNN.csv", sep=';')

x = df.iloc[:, 4:]
y = df.iloc[:, 0:4]
num_features = df.shape[1] - 4
print(df.isnull().any())

# Standardize the input features
scaler = StandardScaler()

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

x = scaler.fit_transform(x)

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





# Create a KNN regressor with K=3
knn1 = KNeighborsRegressor(n_neighbors=69)
knn2 = KNeighborsRegressor(n_neighbors=69)
knn3 = KNeighborsRegressor(n_neighbors=69)
knn4 = KNeighborsRegressor(n_neighbors=69)


# Fit the model to the training data
knn1.fit(x_train, NN_train)
knn2.fit(x_train, GR_train)
knn3.fit(x_train, NI_train)
knn4.fit(x_train, FI_train)

NNpred = knn1.predict(x_test)
GRpred = knn2.predict(x_test)
NIpred = knn3.predict(x_test)
FIpred = knn4.predict(x_test)




filename = r"C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\predicted_values_KNN_IS.csv"
y_test_list = y_test.values.tolist()
with open(filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(NNpred)):
        predicted_values = [NNpred[i], GRpred[i], NIpred[i], FIpred[i]]
        true_values = y_test_list[i]
        print("Predicted:", [f"{value:.3f}" for value in predicted_values], "True:", true_values)

        combined_values = []
        combined_values.extend(predicted_values)
        combined_values.extend(true_values)
        writer.writerow(combined_values)


df = pd.read_csv(r"C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\testsetKNN.csv", sep=';')


#df['EIG_ratio'] = df['EIG_ratio'].apply(remove_complex_parts)



x = df.iloc[:, 4:]
y_test = df.iloc[:, 0:4]
print(df.isnull().any())



# Apply the scaler to the input features
x = scaler.transform(x)

NN_test = y_test.iloc[:, 0]
GR_test = y_test.iloc[:, 1]
NI_test = y_test.iloc[:, 2]
FI_test = y_test.iloc[:, 3]

NNpred = knn1.predict(x)
GRpred = knn2.predict(x)
NIpred = knn3.predict(x)
FIpred = knn4.predict(x)




filename = r"C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\predicted_values_KNN_OS.csv"
y_test_list = y_test.values.tolist()
with open(filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(NNpred)):
        predicted_values = [NNpred[i], GRpred[i], NIpred[i], FIpred[i]]
        true_values = y_test_list[i]
        print("Predicted:", [f"{value:.3f}" for value in predicted_values], "True:", true_values)

        combined_values = []
        combined_values.extend(predicted_values)
        combined_values.extend(true_values)
        writer.writerow(combined_values)