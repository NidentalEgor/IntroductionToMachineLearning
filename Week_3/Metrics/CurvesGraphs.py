import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

def GetMaxPrecision(recall_bound, column_name):
    precision, recall, thresholds = precision_recall_curve(
        data['true'],
        data[column_name])
    
    # print(type(precision))
    # print(precision.shape)

    max_precision = 0.0
    for i in range(0, precision.shape[0]):
        if recall[i] > 0.7:
            if precision[i] > max_precision:
                max_precision = precision[i]
    
    return max_precision

data = pd.read_csv("scores.csv")
# print(data.head())
print("\nAuc scores:")

score_logreg = roc_auc_score(data['true'], data['score_logreg'])
print("score_logreg = ", score_logreg)

score_svm = roc_auc_score(data['true'], data['score_svm'])
print("score_svm = ", score_svm)

score_knn = roc_auc_score(data['true'], data['score_knn'])
print("score_knn = ", score_knn)

score_tree = roc_auc_score(data['true'], data['score_tree'])
print("score_tree = ", score_tree)

print("\nMax precision:")
logreg_max = GetMaxPrecision(0.7, "score_logreg")
svm_max = GetMaxPrecision(0.7, "score_svm")
knn_max = GetMaxPrecision(0.7, "score_knn")
tree_max = GetMaxPrecision(0.7, "score_tree")

maxes = {'logreg_max' : logreg_max,
        'svm_max' : svm_max,
        'knn_max' : knn_max,
        'tree_max' : tree_max}

print("Max precision:", max(maxes, key = maxes.get))
