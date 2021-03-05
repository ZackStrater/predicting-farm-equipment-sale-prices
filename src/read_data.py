

import pandas as pd
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from loss_model import rmsle
random_state =100


df = pd.read_csv('../data/train.csv')


y = df.pop('SalePrice')
X = df[['YearMade']]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=random_state)
n_folds=10
kf = KFold(n_splits=n_folds, random_state=random_state)

errors = []

for train, test in kf.split(X):
    model = LinearRegression()
    X_train = X.values[train]
    X_test = X.values[test]
    y_train = y.values[train]
    y_test = y.values[test]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error = rmsle(y_test, y_pred)
    errors.append(error)

print(errors)
print(np.mean(np.array(errors)))