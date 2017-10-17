import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('titanic.csv', index_col='PassengerId')

usefull_data = pd.concat(
    [data['Pclass'], data['Fare'], data['Age'], data['Sex'], data['Survived']],
    axis=1,
    keys=['Pclass', 'Fare', 'Age','Sex','Survived'])

print(usefull_data.head(6))

#usefull_data = usefull_data[pd.isnull(usefull_data.Age)]
usefull_data_without_nans = usefull_data.dropna(
    subset = ['Pclass', 'Fare', 'Age','Sex','Survived'])
print(usefull_data_without_nans.head(6))

mapping = {'male' : 1, 'female' : 0}
usefull_data_without_nans = usefull_data_without_nans.replace({'Sex' : mapping})
#usefull_data_without_nans['Sex'] = usefull_data_without_nans['Sex'].astype('category')

print('WOWOWOWOWOWOW',usefull_data_without_nans)

y = usefull_data_without_nans['Survived'].as_matrix()
print(y)

clf = DecisionTreeClassifier(random_state=241)
df = usefull_data_without_nans[['Pclass', 'Fare', 'Age','Sex']]
#print(df)
clf.fit(df.as_matrix(), y)

importances = clf.feature_importances_
print(importances)

#print('Hello!')
#
#X = np.array([[1, 2], [3, 4], [5, 6]])
#y = np.array([0, 1, 0])
#clf = DecisionTreeClassifier()
#clf.fit(X, y)
#
#print(np.isnan(X))
#
#importances = clf.feature_importances_
#print(importances)