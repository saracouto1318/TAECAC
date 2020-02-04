import numpy as np 
import pandas 
import matplotlib.pyplot as plt 
import os, os.path
from math import radians, cos, sin, asin, sqrt
from sklearn.decomposition import PCA
from matplotlib import gridspec 
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn import metrics
from sklearn.preprocessing import normalize, StandardScaler

DIR = 'Datasets'
numFiles = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) - 1

n = 1

while n <= numFiles:
    col_names = ['Classes', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'SMELLS', 'HASS','CHANGES', 'HASC']
    # load dataset
    fileName = DIR + "/" + str(n) + ".csv"
    pima = pandas.read_csv(fileName, header=0, names=col_names)
    pima.drop(['Classes', 'SMELLS', 'HASS','CHANGES', 'HASC'], axis = 1, inplace = True)  

    
    array = []
    totalValue = len(pima.S1)
    i = 0
    
    while i < (len(col_names)-5):
        counter = totalValue
        j = 0
        string = 'S' + str(i+1) 
        while j < (len(col_names)-5):
            stringNew = 'S' + str(j+1)
            if i == j:
                array.append(0)
            else:       
                if pima[stringNew][j] == 1 and pima[string][j] == 1:
                    counter -= 1  
                array.append(counter)       
            j += 1   
        i += 1  
          
    matrix = []

    i = 0
    while i < len(array):
        matrix.append([array[i], array[i+1], array[i+2], array[i+3], array[i+4], array[i+5], array[i+6], array[i+7], array[i+8], array[i+9], array[i+10], array[i+11], array[i+12], array[i+13], array[i+14], array[i+15], array[i+16], array[i+17], array[i+18], array[i+19], array[i+20], array[i+21], array[i+22], array[i+23], array[i+24], array[i+25], array[i+26], array[i+27], array[i+28], array[i+29], array[i+30], array[i+31], array[i+32], array[i+33], array[i+34]])
        i += 35 
        
    distances = np.array(matrix)

    X = distances
    pca = PCA(n_components=2)
    X2d = pca.fit_transform(X) 

    rads = np.radians(X2d)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='haversine')
    cluster_labels = clusterer.fit_predict(X2d)
    n_noise_ = list(cluster_labels).count(-1)

    core_samples_mask = np.zeros_like(cluster_labels, dtype=bool)
    core_samples_mask[clusterer.core_sample_indices_] = True
    labels = cluster_labels
    print(labels)

    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title('number of clusters: %d' %n_clusters_)
    plt.show() 
    plt.savefig('Results/DBSCAN/dbscan'+str(n)+'.png')
    n += 1