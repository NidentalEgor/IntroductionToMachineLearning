from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
# from itertools import izip


newsgroups = datasets.fetch_20newsgroups(
    subset='all', 
    categories=['alt.atheism','sci.space'])

#pd.DataFrame(newsgroups.data).to_csv("data.csv")
#pd.DataFrame(newsgroups.target).to_csv("target.csv")

#data = pd.read_csv("data.csv", header=None)
#target = pd.read_csv("target.csv", header=None)
#v.fit_transform(data.as_matrix().tolist)

v = TfidfVectorizer()
X = v.fit_transform(newsgroups.data)
#y = v.fit_transform(newsgroups.target)

#feature_mapping = v.get_feature_names()

grid = {'C': np.power(10.0, np.arange(-5, 6))}

cv = KFold(
    n_splits = 5,
    shuffle = True,
    random_state = 241)
clf = SVC(
    kernel = 'linear',
    random_state = 241)

gs = GridSearchCV(
    clf,
    grid,
    scoring = 'accuracy',
    cv = cv)

#print ("Before gs.fit()")
# gs.fit(
#    X,
#    newsgroups.target)
#print ("After gs.fit()")

# #for score in gs.grid_scores_:
# for score in gs.cv_results_:
#    print(
#        "mean_validation_score =",
#        # оценка качества по кросс-валидации
#        score.mean_validation_score)
#    print(
#        "parameters =",
#        # значения параметров
#        score.parameters)

# mean_validation_score = 0.552631578947
# parameters = {'C': 1.0000000000000001e-05}
# mean_validation_score = 0.552631578947
# parameters = {'C': 0.0001}
# mean_validation_score = 0.552631578947
# parameters = {'C': 0.001}
# mean_validation_score = 0.552631578947
# parameters = {'C': 0.01}
# mean_validation_score = 0.950167973124
# parameters = {'C': 0.10000000000000001}
# mean_validation_score = 0.993281075028
# parameters = {'C': 1.0}
# mean_validation_score = 0.993281075028
# parameters = {'C': 10.0}
# mean_validation_score = 0.993281075028
# parameters = {'C': 100.0}
# mean_validation_score = 0.993281075028
# parameters = {'C': 1000.0}
# mean_validation_score = 0.993281075028
# parameters = {'C': 10000.0}
# mean_validation_score = 0.993281075028
# parameters = {'C': 100000.0}

clf1 = SVC(
    kernel = 'linear',
    random_state = 241,
    C = 1.0)

clf1.fit(
    X,
    newsgroups.target)
print(type(clf1.coef_))
#print(type(clf1.coef_.toarray()))
csr_m = clf1.coef_
#print("csr_m", csr_m)
#print("csr_m.sort_indices()", csr_m.sorted_indices())
ada = np.absolute(csr_m.toarray())
#print("np.absolute(csr_m)", ada)

#print(csr_m.toarray())
#print(np.absolute(csr_m.toarray()))
#pd.DataFrame(clf1.coef_).to_csv("coef.csv")

#temp = np.argpartition(-clf1.coef_.toarray(), 10)
#result_args = temp[0:10]
#
#temp2 = np.partition(-clf1.coef_.toarray(), 10)
#result = -temp2[0:10]

#print("result_args = ", result_args)
#print("result_args = ", result_args.shape)
#print("result = ", result)
#print("result = ", result.shape)

#for index in result_args:
#    print(feature_mapping[index])
#
#
#print ("Result:")
#for index in result_args:
#    print(feature_mapping[index])

# for a3 in gs.grid_scores_.toarray():
#     print(a3.mean_validation_score)# — оценка качества по кросс-валидации
#     print(a3.parameters)# — значения параметров
#     print(a3)

####################################

#word_indexes = np.argsort(np.abs(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]
#words = [feature_mappings[i] for i in word_indexes]
#
#word = pd.DataFrame(data=v.get_feature_names())
#coef = pd.DataFrame(data=np.abs(np.asarray(clf.coef_.todense()).reshape(-1)))
#data = pd.concat([name, coef], axis=1)
#data.columns = ['word', 'coef']
#data.sort_index(by=['coef'])[-10:]

#print("RES:", data)

###################################

print(type(clf1.coef_))
print(clf1.coef_)

coefs = clf1.coef_

# tuples = izip(coefs.row, coefs.col, coefs.data)
# sorted = sorted(tuples, key=lambda x: (x[0], x[2]))

# print("Sorted",sorted)
# print(type(coefs.indices))
# print(coefs.indices)
# print(coefs.indices.shape)

word = pd.DataFrame(data=v.get_feature_names())
coef = pd.DataFrame(data=np.abs(np.asarray(clf1.coef_.todense()).reshape(-1)))

print(coef)

data = pd.concat([word, coef], axis=1)
data.columns = ['word', 'coef']
top = data.sort_values(by=['coef'])[-10:]

# print(type(top))

top_top = top.sort_values(by=['word'])['word']
# print(top_top)

a = np.argsort(np.abs(np.asarray(clf1.coef_.todense())).reshape(-1))[-10:]
print(a)
feature_names = np.asarray(v.get_feature_names())[a]
print(feature_names)
print(np.sort(feature_names))

word_indexes = np.argsort(np.abs(np.asarray(clf1.coef_.todense())).reshape(-1))[-10:]

words = [v.get_feature_names()[i] for i in word_indexes]

# print(words)
print(np.sort(words))

# print("as array:",np.asarray(clf1.coef_.todense()).reshape(-1))
# weights_abs = np.abs(np.asarray(clf1.coef_.todense()).reshape(-1)[0]) # абсолютные значения весов признаков

# weights_indices_sorted = np.argsort(weights_abs) # индексы весов отсортированных по возрвстанию

# top10_indices = weights_indices_sorted[-10:] # индексы топ10 весов

# feature_names = np.asarray(v.get_feature_names())[top10_indices] # названия признаков соответсвующих этим индексам

# print(" ".join(np.sort(feature_names))) # сортировка по алфавиту

# 22936       sci  1.029307
# 15606     keith  1.097094
# 5776      bible  1.130612
# 21850  religion  1.139081
# 23673       sky  1.180132
# 17802      moon  1.201611
# 5093   atheists  1.249180
# 5088    atheism  1.254690
# 12871       god  1.920379
# 24019     space  2.663165

#mylist = ['sci','keith','bible','religion','sky','moon','','',,'','']

# print(coefs.data)
# print(type(coefs.data))
# print(coefs.data.shape)
# datata = pd.DataFrame(coefs.indices, coefs.data, columns = ['id', 'weight'])
# print(datata)
# a = clf1.coef_.toarray()

# aa = np.absolute(a)
# a1 = aa.argsort()#[-3:][::-1]
# print("a1", a1)

# list_result = []
# for index in range(10):
#     list_result.append(feature_mapping[a1.ravel()[index]])

# list_result.sort()

# print("result:")
# for word in list_result:
#     print(word)