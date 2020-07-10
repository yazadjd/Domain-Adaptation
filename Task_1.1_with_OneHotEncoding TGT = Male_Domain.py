# -*- coding: utf-8 -*-
"""OneHotEncoding of SML_Project2_Male.ipynb

Automatically generated by Colaboratory.

This notebook contains code relevant to running all baselines using one hot encoded features for target domain = MALE for Task 1.1 of SML Project 2 - Sem 1, 2020.
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
from xgboost import XGBRegressor
from operator import itemgetter
from scipy import interpolate

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My Drive/SML_Proj2

male = pd.read_csv("MALE.csv")
female = pd.read_csv("FEMALE.csv")
mixed = pd.read_csv("MIXED.csv")

male_features  = male[['Year',	'FSM',	'VR1 Band',	'VR Band of Student',	'Ethnic group of student',	'School denomination']]
male_scores = male [['Exam Score']]

to_encode = ['Year', 'VR Band of Student', 'Ethnic group of student', 'School denomination']

male_features = pd.get_dummies(male_features, prefix_sep = "__", columns = to_encode)

columns = list(male_features)
print(len(columns))

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

columns = list(female_features)
print(columns)

#Splitting data for female

X_train, female_X_test, y_train, female_Y_test = train_test_split(female_features, female_scores, test_size = 0.2, random_state = 42)
female_X_train, female_X_val, female_Y_train, female_Y_val = train_test_split(X_train, y_train, test_size = 100, random_state = 42)

#Checking distribution using mean of each partition
print(mean(female_Y_test['Exam Score']))
print(mean(female_Y_train['Exam Score']))
print(mean(female_Y_val['Exam Score']))

mixed.drop(mixed[mixed['VR Band of Student'] == 0].index, inplace = True) 
mixed_features  = mixed[['Year',	'FSM',	'VR1 Band',	'VR Band of Student',	'Ethnic group of student',	'School denomination']]
mixed_scores = mixed[['Exam Score']]

mixed_features = pd.get_dummies(mixed_features, prefix_sep = "__", columns = to_encode)

columns = list(mixed_features)
print(columns)

#Splitting data for mixed

X_train, mixed_X_test, y_train, mixed_Y_test = train_test_split(mixed_features, mixed_scores, test_size = 0.2, random_state = 42)
mixed_X_train, mixed_X_val, mixed_Y_train, mixed_Y_val = train_test_split(X_train, y_train, test_size = 100, random_state = 42)

#Checking distribution using mean of each partition
print(mean(mixed_Y_test['Exam Score']))
print(mean(mixed_Y_train['Exam Score']))
print(mean(mixed_Y_val['Exam Score']))

#Reproducible Sampling function to get 100 instances without replacement
def sample(df):
  return df.sample(n = 100, replace = False, random_state = 42)

"""TARGET DOMAIN = MALE

SRCONLY
"""

x_train = pd.concat([female_X_train, mixed_X_train])
y_train = pd.concat([female_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, male_X_val, male_Y_val)
lin_reg(x_train, y_train, male_X_test, male_Y_test)

# random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, male_X_val, male_Y_val, 12)
MLP_Reg(x_train, y_train, male_X_test, male_Y_test, 12)

"""TGTONLY"""

x_train = sample(male_X_train)
y_train = sample(male_Y_train)

lin_reg(x_train, y_train, male_X_val, male_Y_val)
lin_reg(x_train, y_train, male_X_test, male_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, male_X_val, male_Y_val, 17)
MLP_Reg(x_train, y_train, male_X_test, male_Y_test, 17)

"""ALL"""

x_train = pd.concat([sample(male_X_train), female_X_train, mixed_X_train])
y_train = pd.concat([sample(male_Y_train), female_Y_train, mixed_Y_train])

lin_reg(x_train, y_train, male_X_val, male_Y_val)
lin_reg(x_train, y_train, male_X_test, male_Y_test)

#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, male_X_val, male_Y_val, 12)
MLP_Reg(x_train, y_train, male_X_test, male_Y_test, 12)

"""WEIGHTED"""

x_train = pd.concat([sample(male_X_train)]*91, ignore_index=True)
x_train = pd.concat([x_train, x_train.head(54), female_X_train, mixed_X_train], ignore_index = True)
y_train = pd.concat([sample(male_Y_train)]*91, ignore_index=True)
y_train = pd.concat([y_train, y_train.head(54), female_Y_train, mixed_Y_train], ignore_index = True)

lin_reg(x_train, y_train, male_X_val, male_Y_val)
lin_reg(x_train, y_train, male_X_test, male_Y_test)

# MLPRegressor(random_state = 42, max_iter = 10000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
MLP_Reg(x_train, y_train, male_X_val, male_Y_val, 10)
MLP_Reg(x_train, y_train, male_X_test, male_Y_test, 10)

def lin_reg(x_train, y_train, x_val, y_val):
  regr = linear_model.LinearRegression()
  regr.fit(x_train, y_train)
  lin_pred = regr.predict(x_val)
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, lin_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_val, lin_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, lin_pred)))
 # return lin_pred

all = []
for n in range(1,200):
  all.append((MLP_Reg(x_train, y_train, male_X_val, male_Y_val, n), n))
print(min(all, key = itemgetter(0))[1])

"""PRED"""

x_train = pd.concat([female_X_train, mixed_X_train])
y_train = pd.concat([female_Y_train, mixed_Y_train])

predictions = lin_reg(x_train, y_train, male_features, male_scores)
augmented_male_features = male_features.assign(src_preds = predictions)
X_train, male_X_test, y_train, male_Y_test = train_test_split(augmented_male_features, male_scores, test_size = 0.2, random_state = 42)
male_X_train, male_X_val, male_Y_train, male_Y_val = train_test_split(X_train, y_train, test_size = 100, random_state = 42)
x_train = sample(male_X_train)
y_train = sample(male_Y_train)

lin_reg(x_train, y_train, male_X_val, male_Y_val)
lin_reg(x_train, y_train, male_X_test, male_Y_test)

#MLPRegressor(random_state = 42, max_iter = 10000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
predictions = MLP_Reg(x_train, y_train, male_features, male_scores, 12)
augmented_male_features = male_features.assign(src_preds = predictions)
X_train, male_X_test, y_train, male_Y_test = train_test_split(augmented_male_features, male_scores, test_size = 0.2, random_state = 42)
male_X_train, male_X_val, male_Y_train, male_Y_val = train_test_split(X_train, y_train, test_size = 100, random_state = 42)
x_train = sample(male_X_train)
y_train = sample(male_Y_train)

male_X_train.head()

# MLPRegressor(random_state = 42, max_iter = 10000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.0018)
MLP_Reg(x_train, y_train, male_X_val, male_Y_val, 6)
MLP_Reg(x_train, y_train, male_X_test, male_Y_test, 6)

"""LININT"""

def MLP_Reg(x_train, y_train, x_val, y_val, h):
  regr = MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
  regr.fit(x_train, y_train)
  MLP_pred = regr.predict(x_val)
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, MLP_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_val, MLP_pred), 'h = ', h)
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, MLP_pred)))
  return metrics.mean_squared_error(y_val, MLP_pred)
  #return MLP_pred

x_train = pd.concat([female_X_train, mixed_X_train])
y_train = pd.concat([female_Y_train, mixed_Y_train])
# MLPRegressor(random_state = 42, max_iter = 10000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h)
srconly_preds = MLP_Reg(x_train, y_train, male_X_val, male_Y_val, 12)
srconly_preds_test = MLP_Reg(x_train, y_train, male_X_test, male_Y_test, 12)

x_train = sample(male_X_train)
y_train = sample(male_Y_train)
#MLPRegressor(random_state = 42, max_iter = 1000, activation = 'relu', solver = 'adam', hidden_layer_sizes=h, learning_rate='constant', learning_rate_init= 0.001)
tgtonly_preds = MLP_Reg(x_train, y_train, male_X_val, male_Y_val, 17)
tgtonly_preds_test = MLP_Reg(x_train, y_train, male_X_test, male_Y_test, 17)
data = {'y1' : srconly_preds, 'y2 - y1' : np.absolute(tgtonly_preds - srconly_preds)}
x_train = pd.DataFrame(data)
y_train = male_Y_val
regr = linear_model.LinearRegression(fit_intercept = False) 
regr.fit(x_train, y_train) #Coeff of y1 is approx 1
print(regr.coef_)
data = {'y1' : srconly_preds_test, 'y2 - y1' : np.absolute(tgtonly_preds_test - srconly_preds_test)}
lin_pred = regr.predict(pd.DataFrame(data))
print(metrics.mean_squared_error(male_Y_test, lin_pred))

x_train = pd.concat([female_X_train, mixed_X_train])
y_train = pd.concat([female_Y_train, mixed_Y_train])
srconly_preds = lin_reg(x_train, y_train, male_X_val, male_Y_val)
srconly_preds_test = lin_reg(x_train, y_train, male_X_test, male_Y_test)

x_train = sample(male_X_train)
y_train = sample(male_Y_train)

tgtonly_preds = lin_reg(x_train, y_train, male_X_val, male_Y_val)
tgtonly_preds_test = lin_reg(x_train, y_train, male_X_test, male_Y_test)
#data = {'y1' : srconly_preds, 'y2 - y1' : np.absolute(tgtonly_preds - srconly_preds)}
lst2 = np.absolute(tgtonly_preds - srconly_preds)
x_train = pd.DataFrame(list(zip(srconly_preds, lst2)), columns = ['y1', 'y2 - y1']) 
#x_train = pd.DataFrame(data)
y_train = male_Y_val
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train) #Coeff of y1 is approx 1
print(regr.coef_)
#data = {'y1' : srconly_preds_test, 'y2 - y1' : np.absolute(tgtonly_preds_test - srconly_preds_test)}
lin_pred = regr.predict(pd.DataFrame(pd.DataFrame(list(zip(srconly_preds_test, np.absolute(tgtonly_preds_test - srconly_preds_test))), columns = ['y1', 'y2 - y1'])))
print(metrics.mean_squared_error(male_Y_test, lin_pred))
