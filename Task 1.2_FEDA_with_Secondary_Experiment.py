# -*- coding: utf-8 -*-
"""FEDA with Secondary Experiment.ipynb

Automatically generated by Colaboratory.



This notebook contains code relevant to experiments conducted for Task 1.2 of SML - Project 2, Sem 1, 2020.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statistics import mean
import numpy as np
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from operator import itemgetter
from scipy import interpolate
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My Drive/SML_Proj2

male = pd.read_csv("MALE.csv")
female = pd.read_csv("FEMALE.csv")
mixed = pd.read_csv("MIXED.csv")

male_features  = male[['Year',	'FSM',	'VR1 Band',	'VR Band of Student',	'Ethnic group of student',	'School denomination']]
male_scores = male [['Exam Score']]

male_features.head()

to_encode = ['Year', 'VR Band of Student', 'Ethnic group of student', 'School denomination']

male_features = pd.get_dummies(male_features, prefix_sep = "__", columns = to_encode)

male_features.head()

male_features.shape

feature_list = list(male_features)
zero_data = np.zeros(shape=(3654, 22))
zeroes = pd.DataFrame(zero_data, columns=feature_list)

zeroes.shape

male_features = pd.concat([male_features, male_features, zeroes, zeroes], axis = 1)
male_features.head()

male_features.shape

#Splitting data for male
X_train, male_X_test, y_train, male_Y_test = train_test_split(male_features, male_scores, test_size = 0.2, random_state = 42)
male_X_train, male_X_val, male_Y_train, male_Y_val = train_test_split(X_train, y_train, test_size = 100, random_state = 42)

#Checking distribution using mean of each partition
print(mean(male_Y_test['Exam Score']))
print(mean(male_Y_train['Exam Score']))
print(mean(male_Y_val['Exam Score']))

female_features  = female[['Year',	'FSM',	'VR1 Band',	'VR Band of Student',	'Ethnic group of student',	'School denomination']]
female_scores = female [['Exam Score']]

female_features = pd.get_dummies(female_features, prefix_sep = "__", columns = to_encode)

female_features.shape

feature_list = list(female_features)
zero_data = np.zeros(shape=(4404, 22))
zeroes = pd.DataFrame(zero_data, columns=feature_list)

female_features = pd.concat([female_features, zeroes, female_features, zeroes], axis = 1)
female_features.head()

female_features.shape

#Splitting data for female

X_train, female_X_test, y_train, female_Y_test = train_test_split(female_features, female_scores, test_size = 0.2, random_state = 42)
female_X_train, female_X_val, female_Y_train, female_Y_val = train_test_split(X_train, y_train, test_size = 100, random_state = 42)

#Checking distribution using mean of each partition
print(mean(female_Y_test['Exam Score']))
print(mean(female_Y_train['Exam Score']))
print(mean(female_Y_val['Exam Score']))

mixed.shape

mixed.tail()

mixed.drop(mixed[mixed['VR Band of Student'] == 0].index, inplace = True)
mixed.reset_index(inplace = True)

mixed.tail()

mixed_features  = mixed[['Year',	'FSM',	'VR1 Band',	'VR Band of Student',	'Ethnic group of student',	'School denomination']]
mixed_scores = mixed[['Exam Score']]

mixed_features = pd.get_dummies(mixed_features, prefix_sep = "__", columns = to_encode)

mixed_features.shape

feature_list = list(mixed_features)
zero_data = np.zeros(shape=(7289, 22))
zeroes = pd.DataFrame(zero_data, columns=feature_list)

mixed_features = pd.concat([mixed_features, zeroes, zeroes, mixed_features], axis = 1)
mixed_features.head()

mixed_features.shape

#Splitting data for mixed

X_train, mixed_X_test, y_train, mixed_Y_test = train_test_split(mixed_features, mixed_scores, test_size = 0.2, random_state = 42)
mixed_X_train, mixed_X_val, mixed_Y_train, mixed_Y_val = train_test_split(X_train, y_train, test_size = 100, random_state = 42)

#Checking distribution using mean of each partition
print(mean(mixed_Y_test['Exam Score']))
print(mean(mixed_Y_train['Exam Score']))
print(mean(mixed_Y_val['Exam Score']))

male_X_train.head()

"""FEDA

TARGET DOMAIN = MALE

ALL
"""

def lin_reg(x_train, y_train, x_val, y_val):
  regr = linear_model.LinearRegression()
  regr.fit(x_train, y_train)
  lin_pred = regr.predict(x_val)
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, lin_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_val, lin_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, lin_pred)))
  #return lin_pred

x_train = pd.concat([sample(male_X_train), female_X_train, mixed_X_train])
y_train = pd.concat([sample(male_Y_train), female_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, male_X_val, male_Y_val)
lin_reg(x_train, y_train, male_X_test, male_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, male_X_val, male_Y_val, 14)
MLP_Reg(x_train, y_train, male_X_test, male_Y_test, 14)

#Code to get best value of hyperparameter
all = []
for n in range(1,110):
  all.append((MLP_Reg(x_train, y_train, mixed_X_val, mixed_Y_val, n), n))
print(min(all, key = itemgetter(0))[1])

"""TARGET DOMAIN = FEMALE"""

x_train = pd.concat([sample(female_X_train), male_X_train, mixed_X_train])
y_train = pd.concat([sample(female_Y_train), male_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, female_X_val, female_Y_val)
lin_reg(x_train, y_train, female_X_test, female_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, female_X_val, female_Y_val, 11)
MLP_Reg(x_train, y_train, female_X_test, female_Y_test, 11)

"""TARGET DOMAIN = MIXED"""

x_train = pd.concat([sample(mixed_X_train), male_X_train, female_X_train])
y_train = pd.concat([sample(mixed_Y_train), male_Y_train, female_Y_train])

lin_reg(x_train, y_train, mixed_X_val, mixed_Y_val)
lin_reg(x_train, y_train, mixed_X_test, mixed_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, mixed_X_val, mixed_Y_val, 6)
MLP_Reg(x_train, y_train, mixed_X_test, mixed_Y_test, 6)

#Reproducible Sampling function to get n instances without replacement
def sample(df):
  return df.sample(frac = 0.1, replace = False, random_state = 42)

"""Target Training SAMPLES = 0.1"""

def MLP_Reg(x_train, y_train, x_val, y_val, h):
  regr = MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001, alpha = 0.0001)
  regr.fit(x_train, y_train)
  MLP_pred = regr.predict(x_val)
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, MLP_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_val, MLP_pred), 'h = ', h)
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, MLP_pred)))
  return metrics.mean_squared_error(y_val, MLP_pred)
  #return MLP_pred

"""TARGET DOMAIN = MALE

ALL
"""

x_train = pd.concat([sample(male_X_train), female_X_train, mixed_X_train])
y_train = pd.concat([sample(male_Y_train), female_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, male_X_val, male_Y_val)
lin_reg(x_train, y_train, male_X_test, male_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, male_X_val, male_Y_val, 100)
MLP_Reg(x_train, y_train, male_X_test, male_Y_test, 100)

"""TARGET DOMAIN = FEMALE"""

x_train = pd.concat([sample(female_X_train), male_X_train, mixed_X_train])
y_train = pd.concat([sample(female_Y_train), male_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, female_X_val, female_Y_val)
lin_reg(x_train, y_train, female_X_test, female_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, female_X_val, female_Y_val, 100)
MLP_Reg(x_train, y_train, female_X_test, female_Y_test, 100)

"""TARGET DOMAIN = MIXED"""

x_train = pd.concat([sample(mixed_X_train), male_X_train, female_X_train])
y_train = pd.concat([sample(mixed_Y_train), male_Y_train, female_Y_train])

lin_reg(x_train, y_train, mixed_X_val, mixed_Y_val)
lin_reg(x_train, y_train, mixed_X_test, mixed_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, mixed_X_val, mixed_Y_val, 100)
MLP_Reg(x_train, y_train, mixed_X_test, mixed_Y_test, 100)

#Reproducible Sampling function to get n instances without replacement
def sample(df):
  return df.sample(frac = 0.2, replace = False, random_state = 42)

"""Target Training SAMPLES = 0.2

TARGET DOMAIN = MALE

ALL
"""

x_train = pd.concat([sample(male_X_train), female_X_train, mixed_X_train])
y_train = pd.concat([sample(male_Y_train), female_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, male_X_val, male_Y_val)
lin_reg(x_train, y_train, male_X_test, male_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, male_X_val, male_Y_val, 100)
MLP_Reg(x_train, y_train, male_X_test, male_Y_test, 100)

"""TARGET DOMAIN = FEMALE"""

x_train = pd.concat([sample(female_X_train), male_X_train, mixed_X_train])
y_train = pd.concat([sample(female_Y_train), male_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, female_X_val, female_Y_val)
lin_reg(x_train, y_train, female_X_test, female_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, female_X_val, female_Y_val, 100)
MLP_Reg(x_train, y_train, female_X_test, female_Y_test, 100)

"""TARGET DOMAIN = MIXED"""

x_train = pd.concat([sample(mixed_X_train), male_X_train, female_X_train])
y_train = pd.concat([sample(mixed_Y_train), male_Y_train, female_Y_train])

lin_reg(x_train, y_train, mixed_X_val, mixed_Y_val)
lin_reg(x_train, y_train, mixed_X_test, mixed_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, mixed_X_val, mixed_Y_val, 100)
MLP_Reg(x_train, y_train, mixed_X_test, mixed_Y_test, 100)

"""Target Training SAMPLES = ALL

TARGET DOMAIN = MALE

ALL
"""

x_train = pd.concat([male_X_train, female_X_train, mixed_X_train])
y_train = pd.concat([male_Y_train, female_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, male_X_val, male_Y_val)
lin_reg(x_train, y_train, male_X_test, male_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, male_X_val, male_Y_val, 100)
MLP_Reg(x_train, y_train, male_X_test, male_Y_test, 100)

"""TARGET DOMAIN = FEMALE"""

x_train = pd.concat([female_X_train, male_X_train, mixed_X_train])
y_train = pd.concat([female_Y_train, male_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, female_X_val, female_Y_val)
lin_reg(x_train, y_train, female_X_test, female_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, female_X_val, female_Y_val, 100)
MLP_Reg(x_train, y_train, female_X_test, female_Y_test, 100)

"""TARGET DOMAIN = MIXED"""

x_train = pd.concat([mixed_X_train, male_X_train, female_X_train])
y_train = pd.concat([mixed_Y_train, male_Y_train, female_Y_train])

lin_reg(x_train, y_train, mixed_X_val, mixed_Y_val)
lin_reg(x_train, y_train, mixed_X_test, mixed_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, mixed_X_val, mixed_Y_val, 100)
MLP_Reg(x_train, y_train, mixed_X_test, mixed_Y_test, 100)

#Reproducible Sampling function to get n instances without replacement
def sample(df):
  return df.sample(frac = 0.5, replace = False, random_state = 42)

"""Target Training SAMPLES = 0.5

TARGET DOMAIN = MALE

ALL
"""

x_train = pd.concat([sample(male_X_train), female_X_train, mixed_X_train])
y_train = pd.concat([sample(male_Y_train), female_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, male_X_val, male_Y_val)
lin_reg(x_train, y_train, male_X_test, male_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, male_X_val, male_Y_val, 100)
MLP_Reg(x_train, y_train, male_X_test, male_Y_test, 100)

"""TARGET DOMAIN = FEMALE"""

x_train = pd.concat([sample(female_X_train), male_X_train, mixed_X_train])
y_train = pd.concat([sample(female_Y_train), male_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, female_X_val, female_Y_val)
lin_reg(x_train, y_train, female_X_test, female_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, female_X_val, female_Y_val, 100)
MLP_Reg(x_train, y_train, female_X_test, female_Y_test, 100)

"""TARGET DOMAIN = MIXED"""

x_train = pd.concat([sample(mixed_X_train), male_X_train, female_X_train])
y_train = pd.concat([sample(mixed_Y_train), male_Y_train, female_Y_train])

lin_reg(x_train, y_train, mixed_X_val, mixed_Y_val)
lin_reg(x_train, y_train, mixed_X_test, mixed_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, mixed_X_val, mixed_Y_val, 100)
MLP_Reg(x_train, y_train, mixed_X_test, mixed_Y_test, 100)

#Reproducible Sampling function to get n instances without replacement
def sample(df):
  return df.sample(frac = 0.75, replace = False, random_state = 42)

"""Target Training SAMPLES = 0.75

TARGET DOMAIN = MALE

ALL
"""

x_train = pd.concat([sample(male_X_train), female_X_train, mixed_X_train])
y_train = pd.concat([sample(male_Y_train), female_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, male_X_val, male_Y_val)
lin_reg(x_train, y_train, male_X_test, male_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, male_X_val, male_Y_val, 100)
MLP_Reg(x_train, y_train, male_X_test, male_Y_test, 100)

"""TARGET DOMAIN = FEMALE"""

x_train = pd.concat([sample(female_X_train), male_X_train, mixed_X_train])
y_train = pd.concat([sample(female_Y_train), male_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, female_X_val, female_Y_val)
lin_reg(x_train, y_train, female_X_test, female_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, female_X_val, female_Y_val, 100)
MLP_Reg(x_train, y_train, female_X_test, female_Y_test, 100)

"""TARGET DOMAIN = MIXED"""

x_train = pd.concat([sample(mixed_X_train), male_X_train, female_X_train])
y_train = pd.concat([sample(mixed_Y_train), male_Y_train, female_Y_train])

lin_reg(x_train, y_train, mixed_X_val, mixed_Y_val)
lin_reg(x_train, y_train, mixed_X_test, mixed_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, mixed_X_val, mixed_Y_val, 100)
MLP_Reg(x_train, y_train, mixed_X_test, mixed_Y_test, 100)
