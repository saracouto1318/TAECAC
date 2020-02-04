import pandas
import numpy
import matplotlib.pyplot as plt
import os, os.path
import scipy
import pylab
from scipy.cluster.hierarchy import dendrogram, linkage

DIR = 'Datasets'
numFiles = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) - 1

n = 1
while n <= numFiles:
    names = ['Classes', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'SMELLS', 'HASS','CHANGES', 'HASC']
    fileName = DIR + "/" + str(n) + ".csv"
    data = pandas.read_csv(fileName, header=0, names=names)
    data.drop(['Classes', 'SMELLS', 'HASS','CHANGES', 'HASC'], axis = 1, inplace = True)  
    
    correlations = data.corr()

    i = 0
    while i < (len(correlations)):
        value = -1
        j = 0
        string = 'S' + str(i+1)
        while j < (len(correlations[string])):
            if numpy.isnan(correlations[string][j]):
                if i == j:
                    value = 1
                else:
                    value = 0    
                correlations[string][j] = value
            j += 1
        i += 1
    
    corr = correlations

    Z = linkage(corr, 'ward')
    plt.figure(figsize=(25, 10))
    labelsize=20
    ticksize=15
    index = str(n)
    plt.title('Hierarchical Clustering Dendrogram for Dataset #'+index, fontsize=labelsize)
    plt.xlabel('Smells', fontsize=labelsize)
    plt.ylabel('Distance', fontsize=labelsize)
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        labels = corr.columns
    )
    plt.axhline(y=1, color='r', linestyle='--')
    pylab.yticks(fontsize=ticksize)
    pylab.xticks(rotation=-90, fontsize=ticksize)
    plt.savefig('Results/HierarchicalClustering/dendrogram_'+index+'.png')
    n += 1