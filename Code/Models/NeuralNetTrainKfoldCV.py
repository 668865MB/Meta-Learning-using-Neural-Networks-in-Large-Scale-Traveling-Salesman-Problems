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
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import scipy.stats as stats
import pandas as pd
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

df = pd.read_csv(r"C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\Train9000.csv", sep=';')

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

# Calculate the correlation matrix
correlation_matrix = x.corr()
correlation_matrix.to_csv(r"C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\correlation_matrix.csv", index=True)
# Assuming 'correlation_matrix' contains your correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.xlabel('Features')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


# Standardize the input features
scaler = StandardScaler()

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

x = scaler.fit_transform(x)

# Set the number of folds (K)
k_folds = 5

# Initialize the K-fold cross-validator
kf = KFold(n_splits=k_folds, shuffle=True)

# Set up the hyperparameters grid for Grid Search
param_grid = {
    'lambda_val': [0.01, 0.001, 0.0001,0],
    'dropout_rate': [0.1, 0.3, 0.5,0],
    'batch_size': [12, 24, 36],
    'learning_rate': [0.01, 0.001, 0.0001]
}


# Generate all parameter combinations
best_spearman = -1

ii =0
all_combinations = list(ParameterGrid(param_grid))
j = len(all_combinations)
# Perform K-fold cross-validation

# Load progress if it exists
progress_file = r"C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\progress.pkl"
if os.path.exists(progress_file):
    with open(progress_file, 'rb') as f:
        progress = pickle.load(f)
else:
    progress = {'current_iteration': 0, 'best_spearman': -1}

# Continue from the last iteration
current_iteration = progress['current_iteration']
best_spearman = progress['best_spearman']


for ii in range(current_iteration,len(all_combinations)):

    combination = all_combinations[ii]
    batch_size = combination['batch_size'] 
    dropout_rate = combination['dropout_rate'] 
    lambda_val = combination['lambda_val'] 
    learning_rate = combination['learning_rate'] 



    model1 = keras.Sequential()
    model1.add(keras.Input(shape=(num_features,)))
    model1.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val))) #,kernel_regularizer=tf.keras.regularizers.l2(0.01)
    model1.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
    model1.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val)))
    model1.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
    model1.add(layers.Dense(1, activation=tanh_tilde))

    model2 = keras.Sequential()
    model2.add(keras.Input(shape=(num_features,)))
    model2.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val)))
    model2.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
    model2.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val)))
    model2.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
    model2.add(layers.Dense(1, activation=tanh_tilde))

    model3 = keras.Sequential()
    model3.add(keras.Input(shape=(num_features,)))
    model3.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val)))
    model3.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
    model3.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val)))
    model3.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
    model3.add(layers.Dense(1, activation=tanh_tilde))

    model4 = keras.Sequential()
    model4.add(keras.Input(shape=(num_features,)))
    model4.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val)))
    model4.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
    model4.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val)))
    model4.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
    model4.add(layers.Dense(1, activation=tanh_tilde))

    model1.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=1.0), loss="mse")
    model2.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=1.0), loss="mse")
    model3.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=1.0), loss="mse")
    model4.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=1.0), loss="mse")

    #log_dir = "logs/"
    #tensorboard_callback = TensorBoard(log_dir=log_dir)
    #hist = model.fit(x_train, [NN_train, GR_train, NI_train, FI_train], callbacks=[early_stopping], epochs=1000, validation_data=(x_test, y_test))

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

        # Define the early stopping callback
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        hist = model1.fit(x_train, NN_train, epochs=500, batch_size=batch_size, verbose=0, callbacks=[early_stopping])
        print("model 1 complete")
        hist = model2.fit(x_train, GR_train,epochs=500, batch_size=batch_size,verbose=0, callbacks=[early_stopping])
        print("model 2 complete")
        hist = model3.fit(x_train, NI_train,epochs=500, batch_size=batch_size,verbose=0, callbacks=[early_stopping])
        print("model 3 complete")
        hist = model4.fit(x_train, FI_train,epochs=500, batch_size=batch_size,verbose=0, callbacks=[early_stopping])
        print("model 4 complete")

        NNpred = model1.predict(x_test, verbose=0)
        GRpred = model2.predict(x_test, verbose=0)
        NIpred = model3.predict(x_test, verbose=0)
        FIpred = model4.predict(x_test, verbose=0)


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
    if mean_spearman > best_spearman:
        best_batch_size = batch_size
        best_dropout_rate = dropout_rate 
        best_lambda_val = lambda_val 
        best_learning_rate = learning_rate
        best_spearman = mean_spearman
    
    result = [batch_size,dropout_rate,lambda_val,learning_rate,mean_spearman]
    with open(r'C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\KFOLD', 'a', newline='') as csvfile1:
        writer1 = csv.writer(csvfile1)
        writer1.writerow(result)
    
    percentage = ii / j * 100
    output = "Progress: {:.2f}%".format(percentage)
    print(output) 
    print("Combo: {:.2f}/{:.2f}".format(ii,j))

    # Save progress
    progress['current_iteration'] = ii
    progress['best_spearman'] = best_spearman
    with open(progress_file, 'wb') as f:
        pickle.dump(progress, f)
   
        

# Print the correlation coefficient
print("Spearman's correlation coefficient:", best_spearman)
print("batch size:", best_batch_size)
print("dropout rate:", best_dropout_rate)
print("lambda:", best_lambda_val)
print("learning rate:", best_learning_rate)


