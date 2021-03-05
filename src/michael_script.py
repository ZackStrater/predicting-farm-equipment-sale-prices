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


df = pd.read_csv('../data/train.csv', low_memory=False)

df['UsageBand'].fillna('other', inplace=True)
df = pd.concat([df.drop('UsageBand', axis=1), pd.get_dummies(df['UsageBand'], prefix='UsageBand')], axis=1)

y = df.pop('SalePrice')
X = df[['YearMade',
        'UsageBand_High',
        'UsageBand_Medium',
        'UsageBand_Low']]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
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

