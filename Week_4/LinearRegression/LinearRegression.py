from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
import scipy

class Processer:
    
    def __init__(self, min_num):
        self.dict_vectorizer = DictVectorizer()
        self.vectorizer = TfidfVectorizer(min_df=min_num)


    def ProcessData(self, data_frame, train_or_test):
        data_frame['FullDescription'] = \
            data_frame['FullDescription'].str.lower()
        data_frame['FullDescription'] = \
            data_frame['FullDescription'].replace(
                '[^a-zA-Z0-9]',
                ' ',
                regex = True)    

        texts = None
        if train_or_test == "train":
            texts = self.vectorizer.fit_transform(data_frame['FullDescription'])
        else:
            texts = self.vectorizer.transform(data_frame['FullDescription'])

        data_frame['LocationNormalized'].fillna('nan', inplace=True)
        data_frame['ContractTime'].fillna('nan', inplace=True)

        X_train_categ = None
        if train_or_test == "train":
            X_train_categ = \
                self.dict_vectorizer.fit_transform(
                    data_frame[['LocationNormalized', 'ContractTime']].to_dict('records'))
        else:
            X_train_categ = \
                self.dict_vectorizer.transform(
                    data_frame[['LocationNormalized', 'ContractTime']].to_dict('records'))

        res = \
            scipy.sparse.hstack(
                [X_train_categ,
                texts])

        return res


################################

processer = Processer(5)

data_train = pd.read_csv("salary-train.csv")
train_matrix = processer.ProcessData(data_train, "train")
print("Train matrix\n", train_matrix.shape)

data_test = pd.read_csv("salary-test-mini.csv")
test_matrix = processer.ProcessData(data_test, "test")
print("Test matrix\n", test_matrix.shape)

ridge = Ridge(alpha=1.0, random_state=241)
ridge.fit(
    train_matrix,
    data_train['SalaryNormalized'])
print(ridge)
# print(test_matrix)

res = ridge.predict(test_matrix)
print(res)