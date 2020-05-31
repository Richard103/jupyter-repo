# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:35:03 2019

@author: Root
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Read the data
train = pd.read_csv('Data/train.csv', index_col = 'Id')
test = pd.read_csv('Data/test.csv')

# Remove rows with missing target, separate target from predictors
X = train.copy()
X.dropna(axis=0, subset=['Cover_Type'], inplace=True)
y = X.Cover_Type
X.drop(['Cover_Type'], axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, 
                                                      random_state = 0)

# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=150, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

#print("Mean Absolute Error for RandomForestRegressor: " + str(score_dataset(X_train, X_valid, y_train, y_valid)))




# cross-validation

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

#my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()), 
#                       ('model', RandomForestRegressor(n_estimators=100, random_state=0))])
#
#score = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
#
#print("Average MAE score:", score.mean())

def get_score(n_estimators=200):
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()), 
                       ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))])
    score = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    return score.mean()

#results = {}
#for i in [50, 100, 150, 200, 250, 300, 350, 400]:
#    results[i] = get_score(i)
    
# the best n_estimator is 200

#print("Mean Absolute Error for RandomForestRegressor with Cross Validation: " + str(get_score()))

import matplotlib.pyplot as plt

#plt.plot(results.keys(), results.values())
#plt.show

from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

xgb_preds = xgb_model.predict(X_valid)
print("Mean Absolute Error for XGBoosting: " + str(mean_absolute_error(xgb_preds, y_valid)))

X.loc[:, 'Elevation':'Horizontal_Distance_To_Fire_Points'].hist(bins=50, figsize=(20,15))
plt.show