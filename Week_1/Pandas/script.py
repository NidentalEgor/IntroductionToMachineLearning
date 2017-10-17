import pandas as pd
import numpy as np

data = pd.read_csv('titanic.csv', index_col='PassengerId')
#data = pd.read_csv('test.csv', index_col='PassengerId')

#print(data.head())

print((data['Sex'] == 'male').sum(),(data['Sex'] == 'female').sum())

print(round(data['Survived'].sum() * 100 / data.shape[0], 2))

print(round((data['Pclass'] == 1).sum() * 100 / data.shape[0], 2))

print(round(data['Age'].mean(), 2), round(data['Age'].median(), 2))

print(round(data['SibSp'].corr(data['Parch']), 2))

print('Test')
print(data.count())
print(data[data['Name'].str.contains('Miss.')]['Name'].count())
print(data[data['Name'].str.contains('Mrs.')]['Name'].count())
print(data[data['Name'].str.contains('Mr.')]['Name'].count())
print(data[data['Name'].str.contains('Mr.|Mrs.|Miss.')]['Name'].count())
#print(data[data['Name'].str.contains('Mr')]['Name'])

#print(data['Name'])