import pandas as pd
import numpy as np
from sklearn.svm import SVC

df = pd.read_csv('svm-data.csv', header=None)

X = df.iloc[0:,1:3]
y = df.iloc[0:,0:1]
print(X.head())
print(y.head())

svc = SVC(
    C=100000,
    kernel='linear',
    random_state = 241)
svc.fit(X, y.as_matrix().ravel())

print(svc.support_)
print(svc.support_vectors_)

