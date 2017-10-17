import pandas as pd
import numpy as np

data = pd.read_csv('titanic.csv', index_col='PassengerId')

women_names = pd.DataFrame(data[data['Sex'] == 'female'])

names = women_names.Name.tolist()

names_frequency = {}
for name in names:
    if (name.find('Miss.') != -1):
        start = name.find('Miss.')
        end = name.find(' ', start + 6)
        res = name[start + 6: end]
    elif (name.find('Mrs.') != -1 or
            name.find('Dr.') != -1):
        start = name.find('(')
        end = name.find(' ', start + 1)
        res = name[start + 1: end]
    count = names_frequency.get(res, 0)
    names_frequency[res] = count + 1

max = 0
for name,fr in names_frequency.items():
    if (fr > max):
        max = fr
        res = name

f= open('MostPopularWomenName.txt', 'w')
f.write('{}'.format(
    res))