import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

def Train(x_train, y_train, x_test, y_test):
    clf = Perceptron(random_state=241)
    clf.fit(x_train, y_train.ravel())
    predictions = clf.predict(x_test)
    return accuracy_score(y_test, predictions)


data_train = pd.read_csv('perceptron-train.csv',header=None)
x_train = data_train.iloc[0:, 1:3]
y_train = data_train.iloc[0:, 0:1]

print(data_train.head())
print(x_train.head())
print(y_train.head())

data_test = pd.read_csv('perceptron-test.csv',header=None)
x_test = data_test.iloc[0:, 1:3]
y_test = data_test.iloc[0:, 0:1]

score_without_scaling = Train(
    x_train.as_matrix(),
    y_train.as_matrix(),
    x_test.as_matrix(),
    y_test.as_matrix())

print(
    "Before scaling",
    score_without_scaling)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(
    x_train.as_matrix())
print(x_train.head())
#print(x_train_scaled.head())
x_test_scaled = scaler.transform(
    x_test.as_matrix())

score_with_scaling = Train(
    x_train_scaled,
    y_train.as_matrix(),
    x_test_scaled,
    y_test.as_matrix())

print(
    "After scaling",
    score_with_scaling)

print(
    "Diff",
    round(
        score_with_scaling - score_without_scaling,
        3))