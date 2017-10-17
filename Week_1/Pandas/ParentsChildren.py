import pandas as pd
import numpy as np

data = pd.read_csv('titanic.csv', index_col='PassengerId')

f = open('ParentChildren.txt', 'w')
f.write('{}'.format(
    round(data['SibSp'].corr(data['Parch']), 2)))