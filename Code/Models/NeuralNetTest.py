import numpy as np
import random
import csv
import pickle
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from tensorboard import notebook
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout





import pandas as pd
import re

# Assuming 'column_name' is the column you want to convert

def ranknet_loss(y_true, y_pred):
    # Compute the pairwise probabilities
    prob_i_greater_j = tf.sigmoid(y_pred[:, 0] - y_pred[:, 1])
    prob_j_greater_i = tf.sigmoid(y_pred[:, 1] - y_pred[:, 0])

    # Compute the RankNet Loss
    loss = -tf.math.log(prob_i_greater_j) - tf.math.log(prob_j_greater_i)
    return tf.reduce_mean(loss)

def listnet_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    # Convert the true and predicted relevance scores to probability distribution
    y_true = tf.exp(y_true) / tf.reduce_sum(tf.exp(y_true))
    y_pred = tf.exp(y_pred) / tf.reduce_sum(tf.exp(y_pred))

    # Calculate the ListNet loss
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred))
    return loss

def listMLE_loss(y_true, y_pred):
    # Compute the ListMLE loss
    
    # Reshape the true and predicted rankings
    y_true = tf.reshape(y_true, [-1])  # Flatten true rankings
    y_pred = tf.reshape(y_pred, [-1])  # Flatten predicted rankings
    
    # Calculate the probabilities of each item being ranked higher than others
    prob_matrix = tf.expand_dims(y_pred, axis=1) - tf.expand_dims(y_pred, axis=0)
    prob_matrix = tf.sigmoid(prob_matrix)
    
    # Compute the log probabilities and sum them across each ranking
    log_probs = tf.reduce_sum(tf.math.log(tf.reduce_sum(prob_matrix * tf.cast(y_true, dtype=tf.float32), axis=1) + 1e-10))
    
    # Take the negative of the average log probability
    loss = -log_probs / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
    
    return loss


def tanh_tilde(x):
    return 1.5 * (K.tanh(x) + 1) + 1

def remove_complex_parts(value):
    value = str(value)  # Convert to string if not already
    value = value.replace('(', '').replace(')', '')  # Remove brackets
    value = value.replace('-0j', '').replace('+0j', '')  # Remove '-0j' and '+0j'
    return float(value)  # Convert to float

df = pd.read_csv(r"C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\TrainComplete.csv", sep=';')
#df = df.sample(n=10, random_state=42) #randompick10

#df['EIG_ratio'] = df['EIG_ratio'].apply(remove_complex_parts)



x = df.iloc[:, 4:]
y_test = df.iloc[:, 0:4]
num_features = df.shape[1]-4
print(df.isnull().any())

# Standardize the input features
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

x = scaler.fit_transform(x)
#num_classes = 4
#one_hot_targets = keras.utils.to_categorical(y - 1, num_classes)



NN_test = y_test.iloc[:, 0]
GR_test = y_test.iloc[:, 1]
NI_test = y_test.iloc[:, 2]
FI_test = y_test.iloc[:, 3]

dropout_rate = 0.3
lambda_val = 0.001


model1 = keras.Sequential()
model1.add(keras.Input(shape=(num_features,)))
model1.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val))) #,kernel_regularizer=tf.keras.regularizers.l2(0.01)
model1.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
model1.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val)))
model1.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
model1.add(layers.Dense(1, activation=tanh_tilde))

model2 = keras.Sequential()
model2.add(keras.Input(shape=(num_features,)))
model2.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val))) #,kernel_regularizer=tf.keras.regularizers.l2(0.01)
model2.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
model2.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val)))
model2.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
model2.add(layers.Dense(1, activation=tanh_tilde))

model3 = keras.Sequential()
model3.add(keras.Input(shape=(num_features,)))
model3.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val))) #,kernel_regularizer=tf.keras.regularizers.l2(0.01)
model3.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
model3.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val)))
model3.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
model3.add(layers.Dense(1, activation=tanh_tilde))

model4 = keras.Sequential()
model4.add(keras.Input(shape=(num_features,)))
model4.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val))) #,kernel_regularizer=tf.keras.regularizers.l2(0.01)
model4.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
model4.add(layers.Dense(8 * num_features, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(lambda_val)))
model4.add(Dropout(dropout_rate))  # Add dropout layer with a dropout rate of 0.5
model4.add(layers.Dense(1, activation=tanh_tilde))


model1.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, clipnorm=1.0), loss="mse")
model2.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, clipnorm=1.0), loss="mse")
model3.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, clipnorm=1.0), loss="mse")
model4.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, clipnorm=1.0), loss="mse")

log_dir = "logs/"
tensorboard_callback = TensorBoard(log_dir=log_dir)


model1.load_weights("localweights1.h5")
model2.load_weights("localweights2.h5")
model3.load_weights("localweights3.h5")
model4.load_weights("localweights4.h5")

#tensorboard --logdir logs

#notebook.start("--logdir logs")


NNpred = model1.predict(x)
GRpred = model2.predict(x)
NIpred = model3.predict(x)
FIpred = model4.predict(x)

filename = r"C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\predicted_values_setD2.csv" #2=Added,#3=Rotation, #4=Shifted, #1=All
y_test_list = y_test.values.tolist()
with open(filename, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(NNpred)):
        predicted_values = [NNpred[i][0], GRpred[i][0], NIpred[i][0], FIpred[i][0]]
        true_values = y_test_list[i]
        print("Predicted:", [f"{value:.3f}" for value in predicted_values], "True:", true_values)

        combined_values = []
        combined_values.extend(predicted_values)
        combined_values.extend(true_values)
        writer.writerow(combined_values)
