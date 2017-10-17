import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage import img_as_float
import matplotlib.pyplot as plt
import pylab
import math

def MSE(imageA, imageB):
    	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA - imageB) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * 3)
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def PSNR(X, X_new):
    return 20 * math.log(
                    (1) / math.sqrt(MSE(X, X_new)), # 255 ** 3?
                    10)

def GetClusters(labels, X):
    clusters = {}
    n = 0
    for item in labels:
        if item in clusters:
            clusters[item].append(X[n])
        else:
            clusters[item] = [X[n]]
        n +=1
    
    return clusters

def GetAvgImage(labels, X, clusters):
    avgDict = {}
    for k,v in clusters.items():
        avgDict[k] = sum(v)/ float(len(v))

    X_mean = np.ndarray(X.shape)
    i = 0
    for item in labels:
        X_mean[i] = avgDict[labels[i]]
        i += 1
    
    return X_mean

def GetMedImage(labels, X, clusters):
    medDict = {}
    for k,v in clusters.items():
        medDict[k] = np.median(v)
    
    X_med = np.ndarray(X.shape)

    i = 0
    for item in labels:
        X_med[i] = medDict[labels[i]]
        i += 1
    
    return X_med

def main():
    new_img = imread('parrots.jpg')
    # pylab.imshow(new_img)
    image = img_as_float(new_img)
    plt.imshow(image)
    plt.show()

    X = image.reshape((image.shape[0] * image.shape[1], 3))

    for i in range(1,21):
        kmeans = KMeans(
                init='k-means++',
                random_state=241,
                n_clusters=i)

        kmeans.fit(X)    
        labels = kmeans.predict(X)

        print("i =", i)

        clusters = GetClusters(labels, X)

        avg_image = GetAvgImage(labels, X, clusters)
        
        # plt.imshow(avg_image.reshape(image.shape))
        # plt.show()

        psnr_avg = PSNR(image, avg_image.reshape(image.shape))
        print("psnr_avg =",psnr_avg)

        med_image = GetMedImage(labels, X, clusters)

        plt.imshow(med_image.reshape(image.shape))
        plt.show()

        psnr_med = PSNR(image, med_image.reshape(image.shape))
        print("psnr_mean =",psnr_med)

    

if __name__ == "__main__":
    main()

