import pandas as pd
import numpy as np

data = pd.read_csv('titanic.csv', index_col='PassengerId')

f = open('MansAndWomens.txt', 'w')
f.write('{} {}'.format(
    (data['Sex'] == 'male').sum(),
    (data['Sex'] == 'female').sum()))