import pandas as pd
import numpy as np

data = pd.read_csv('titanic.csv', index_col='PassengerId')

f = open('FirstClass.txt', 'w')
f.write('{}'.format(
    round((data['Pclass'] == 1).sum() * 100 / data.shape[0], 2)))

