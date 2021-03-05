

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
random_state = 100


df = pd.read_csv('../data/train.csv')
# median_year = df['YearMade'].median()
# df['YearMade'].replace(1000, int(median_year), inplace=True)
df['auctioneerID'].fillna(1000, inplace=True)
auction_id_mean = df.groupby('auctioneerID')['SalePrice'].mean().to_dict()
df['auctioneer_value'] = df['auctioneerID'].map(auction_id_mean)
print(df.isnull().sum())
y = df.pop('SalePrice')
X = df[['YearMade',
        'auctioneer_value']]


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=random_state)
n_folds=10
kf = KFold(n_splits=n_folds)

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