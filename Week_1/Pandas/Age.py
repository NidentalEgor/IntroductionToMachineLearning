import pandas as pd
import numpy as np

data = pd.read_csv('titanic.csv', index_col='PassengerId')

f = open('Age.txt', 'w')
f.write('{} {}'.format(
    round(data['Age'].mean(), 2),
    round(data['Age'].median(), 2)))