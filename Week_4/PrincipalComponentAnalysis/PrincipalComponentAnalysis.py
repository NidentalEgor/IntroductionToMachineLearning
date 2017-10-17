from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def main():
    close_prices = pd.read_csv("close_prices.csv")
    data = pd.DataFrame(close_prices.iloc[0:,1:31])
    # for i in range(10, 1, -1):
    #     pca = PCA(n_components=i)
    #     pca.fit(data)
    #     # print("pca.explained_variance_ratio_ =",pca.explained_variance_ratio_)
    #     print("i =",i,"sum(pca.explained_variance_ratio_) = ",sum(pca.explained_variance_ratio_),"\n")

    print("data.shape =",data.as_matrix().shape)
    pca = PCA(n_components=10)
    pca.fit(data)
    transformed = pca.transform(data)
    print("transformed.shape =",transformed.shape)

    djia_index = pd.read_csv("djia_index.csv")
    print(np.corrcoef(
        transformed[0:,0:1].transpose(),
        djia_index.iloc[0:,1:2].as_matrix().transpose()))

    print(pca.components_.shape)
    first = pca.components_[0:1,0:]
    first[0] = 1000.
    print("Not Abs:",first)
    #first= abs(first)
    print("Abs:",first)
    print(np.argmax(first))

if __name__ == "__main__":
    main()