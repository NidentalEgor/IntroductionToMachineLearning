import numpy as np
import pandas as pd
import sklearn.metrics as metrics

data = pd.read_csv("classification.csv")

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
for index, row in data.iterrows():
    if row['true'] == 1:
        if row['pred'] == 1:
            true_positives += 1
        else:
            false_negatives += 1
    else:
        if row['pred'] == 1:
            false_positives += 1
        else:
            true_negatives += 1

print("\nMetrics:")
print("true_positives =", true_positives)
print("true_negatives =", true_negatives)
print("false_positives =", false_positives)
print("false_negatives =", false_negatives)

true_col = data.iloc[0:,0:1]
pred_col = data.iloc[0:,1:2]

print("\nScores:")
accuracy_score = metrics.accuracy_score(true_col, pred_col)
print("accuracy_score =", round(accuracy_score, 2))

precision_score = metrics.precision_score(true_col, pred_col)
print("precision_score =", round(precision_score, 2))

recall_score = metrics.recall_score(true_col, pred_col)
print("recall_score =", round(recall_score, 2))

f1_score = metrics.f1_score(true_col, pred_col)
print("f1_score =", round(f1_score, 2))
