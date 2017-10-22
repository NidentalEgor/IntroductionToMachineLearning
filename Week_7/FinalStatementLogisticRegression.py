import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import time
import datetime

def ReadData():
    features = pd.read_csv('./features.csv', index_col='match_id')
    radiant_win = features['radiant_win']

    print("ReadData: X.shape =", features.shape)

    return features, radiant_win

def PrepareData(features):
    # Removing data from future.
    features = features.drop(
        ['barracks_status_dire',
            'barracks_status_radiant',
            'tower_status_dire',
            'tower_status_radiant',
            'radiant_win',
            'duration'],
        axis=1)
    
    ###
    print("EGOR EGOR EOGR EGOR EGOR EGOR EOGR EGOR EGOR EGOR EOGR EGOR ", features.shape)
    # print(features.head())
    # print(features.columns.tolist())
    ###
    
    columns_with_missings = features.shape[0] - features.count(axis=0)
    # This is columns with missings.
    print(columns_with_missings[columns_with_missings > 0])

    # TODO: try to replace with large value.
    features.fillna(0, inplace = True)

    return features#, radiant_win

def DropCategorialFeatures(features):
    features = features.drop(
        ['lobby_type',
            'r1_hero',
            'r2_hero',
            'r3_hero',
            'r4_hero',
            'r5_hero',
            'd1_hero',
            'd2_hero',
            'd3_hero',
            'd4_hero',
            'd5_hero'],
        axis=1)

    return features

def DoRegression(X_train, y_train):
    kfold =  KFold(
        n_splits = 5,
        shuffle = True,
        random_state = 241)

    iter = 1
    best_c = 0.000001
    max_score = 0.
    for c in np.power(10.0, np.arange(-5, 6)):
        start_time = datetime.datetime.now()
        score = cross_val_score(
                estimator = LogisticRegression(
                    C=c,
                    random_state = 241),
                X=X_train,
                y=y_train,
                scoring=make_scorer(roc_auc_score),
                cv=kfold)

        print("\nC =", c,"Time elapsed:",datetime.datetime.now() - start_time)
        print("\nIter =",iter,"score =",score.mean(),"\n")

        if score.mean() > max_score:
            best_c = c
            max_score = score.mean()

        iter+=1
    
    print("Best score =",max_score,", best c =",best_c)

def GetHeroesAmount(X):
    heroes_values = np.unique(X[['r1_hero',
            'r2_hero',
            'r3_hero',
            'r4_hero',
            'r5_hero',
            'd1_hero',
            'd2_hero',
            'd3_hero',
            'd4_hero',
            'd5_hero']].values)

    print("Heroes", heroes_values)
    print("Heroes amount =", heroes_values.argmax())
    print("Heroes amount =", heroes_values.shape[0])
    return heroes_values.shape[0]

def BagOfWords(X, N):
    print("Before X.shape =",X.shape)
    categorial_features_names = [
            'r1_hero',
            'r2_hero',
            'r3_hero',
            'r4_hero',
            'r5_hero',
            'd1_hero',
            'd2_hero',
            'd3_hero',
            'd4_hero',
            'd5_hero']
    categorial_features = X[categorial_features_names]
    print("categorial_features.shape = ", categorial_features.shape)

    X_pick = np.zeros((X.shape[0], N))
    print("X_pick.shape =",X_pick.shape)
    print("type(X_pick) =",type(X_pick))

    for i, match_id in enumerate(X.index):
        for p in range(5):
            X_pick[i, X.ix[match_id, 'r%d_hero' % (p+1)]-5] = 1
            X_pick[i, X.ix[match_id, 'd%d_hero' % (p+1)]-5] = -1

    X = X.drop(
            categorial_features_names,
            axis=1)
    print("135",X.shape)
    # result = pd.concat([X, pd.DataFrame(X_pick)], axis = 1, ignore_index=True)
    result = pd.DataFrame(np.concatenate((X.as_matrix(), X_pick), axis=1))

    # i = 0
    # for name in categorial_features:
    #     X[name] = X_pick[:,i]
    #     i+=1

    result.to_csv("categorial_features.csv")
    print("After X.shape =",result.shape)
    return X
    
def ReaTestdData():
    features = pd.read_csv('./features_test.csv', index_col='match_id')
    print("ReadData: X.shape =", features.shape)
    features.fillna(0, inplace = True)

    return features

def DoFinalPredict(X_train, y_train, X_test):
    print("X_train.shape =",X_train.shape)
    print("y_train.shape =", y_train.shape)
    print("X_test.shape =", X_test.shape)
    logistic_regression = LogisticRegression(
            C=0.01,
            random_state = 241)
    logistic_regression.fit(X_train, y_train)
    result = logistic_regression.predict_proba(X_test);
    print("result.shape =",result.shape)
    return result[:,1]

def main():
    X_original, y = ReadData()
    X_test = ReaTestdData()

    X = PrepareData(X_original)

    GetHeroesAmount(X)

    scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X.as_matrix())
    # print(X_scaled)

    # DoRegression(X_scaled, y)
    # Best score = 0.653792271116 , best c = 10.0

    X_without_categorial_features = DropCategorialFeatures(X)
    X_without_categorial_features_scaled = scaler.fit_transform(
            X_without_categorial_features.as_matrix())

    X_test_without_categorial_features = DropCategorialFeatures(X_test)
    X_test_without_categorial_features_scaled = scaler.fit_transform(
            X_test_without_categorial_features.as_matrix())
    # DoRegression(X_without_categorial_features_scaled, y)
    # Best score = 0.65396795402 , best c = 0.01

    
    # X_with_bag_of_words = BagOfWords(X, GetHeroesAmount(X))
    # DoRegression(
    #         scaler.fit_transform(
    #             X_with_bag_of_words.as_matrix()),
    #         y)
    # Best score = 0.653813650883 , best c = 0.01

    result = DoFinalPredict(
            X_without_categorial_features_scaled,
            y,
            X_test_without_categorial_features_scaled)

    print("Min predict result =", result.min())
    print("Max predict result =", result.max())

    print("X_test_without_categorial_features_scaled.shape =", X_test_without_categorial_features_scaled.shape)
    match_id = X_test.index.values
    print(match_id)
    print("match_id.shape =", match_id.shape)
    print("result.shape =", result.shape)
    result = result.reshape((result.shape[0],1))
    match_id = np.around(match_id.reshape((result.shape[0],1)),decimals = 1)
    final_result = np.concatenate(
            (match_id,
            result),
            axis = 1)
    # final_result.tofile("result.csv", sep="\n")

    # with open('result.csv', 'w') as f:
    #     for line in final_result:
    #         print(type(line))
    #         print(line)
    #         f.write(str(line[0] + line[1] + "\n"))

    np.savetxt('result.csv', final_result, fmt='%d, %lf')

if __name__ == "__main__":
    main()