from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from numpy.linalg import norm
from math import exp

class GradientMethod:
    def __init__(self, X, y):
        self.X = X
        self.y = y


    def CalcExp(self, X, y, w, i):
        # print("CalcExp i =", i)
        # print("CalcExp X.shape =", X.shape)
        return 1 - 1 / \
            (1 + exp(-y[i] * (w[0] * X[i][0] + w[1] * X[i][1])))


    def CalcSum(self, X, y, w, w_index):
        sum = 0.0
        for i in range(0, len(y)):
            # print("CalcSum X[i][w_index] =",X[i + 1][w_index + 1])
            # print("CalcSum X.shape =", X.shape)
            # print("CalcSum iter =", i)
            # print("CalcSum w_index =", w_index)

            sum += y[i] * X[i][w_index] * self.CalcExp(X, y, w, i)
        
        sum /= len(y)
        sum *= self.h
        sum -= self.h * self.c * w[w_index]

        # print("CalcSum sum =", sum)
        return sum


    def RecalcW(self, X, y, cur_w):
        # print("SHAPE =", cur_w.shape)
        new_w = cur_w.copy()

        # print("RecalcW cur_w =",cur_w)

        new_w[0] += self.CalcSum(X, y, cur_w, 0)
        new_w[1] += self.CalcSum(X, y, cur_w, 1)
        
        # print("RecalcW new_w =",new_w)
        
        return new_w


    def Do(self, max_iter, h, w0, c):
        self.max_iter = max_iter
        self.h = h
        self.w0 = w0
        self.c = c

        last_w0 = w0;
        cond = True
        i = 0
        while cond:
            new_w0 = self.RecalcW(self.X,self.y,last_w0)
            # print("new_w0",new_w0)
            # print("last_w0",last_w0)
            i += 1
            cur_norm = norm(last_w0-new_w0)

            # print(cur_norm)
            
            cond = i < max_iter and cur_norm > 1e-5
            last_w0 = new_w0
        
        self.w = last_w0

    def Predict(self, X):
        print("Predict X.shape =", X.shape)

        res = np.ndarray(
                (X.shape[0], 1))
        
        for i in range(0, X.shape[0]):
            res[i] = 1 / \
                (1 + exp(-self.w[0] * X[i][0] - self.w[1] * X[i][1]))

        return res
        

def main():
    data = pd.read_csv("data-logistic.csv", header=None)

    X = data.iloc[0:,1:3]
    y = data.iloc[0:,0:1]

    gm = GradientMethod(
        X.as_matrix(),
        y.as_matrix())
    
    gm.Do(
        10000,
        0.1,
        np.ndarray(
            (2,1),
            buffer=np.array([0,0])),
        0)


    res_c_0 = roc_auc_score(y, gm.Predict(X.as_matrix()))
    print(round(res_c_0, 3))

    gm.Do(
        10000,
        0.1,
        np.ndarray(
            (2,1),
            buffer=np.array([0,0])),
        10)

    res_c_10 = roc_auc_score(y, gm.Predict(X.as_matrix()))
    print(round(res_c_10, 3))


if __name__ == "__main__":
    main()