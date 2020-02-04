import matplotlib.pyplot as plt
import os, os.path
import numpy
import csv
import pandas
import scipy.cluster.hierarchy as sch

DIR = 'Datasets'
numFiles = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) - 1
correlations = []

def plot_corr(df,size,n):

    # Compute the correlation matrix for the received dataframe
    correlations = df.corr()
    
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
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(correlations, cmap='RdYlGn')
    plt.xticks(range(len(correlations.columns)), correlations.columns, rotation=90)
    plt.yticks(range(len(correlations.columns)), correlations.columns)
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)
    plt.savefig('Results/Correlation/correlation'+str(n)+'.png')
    
    newFileName = "Processing/Smells" + str(n) + ".csv"
    with open(newFileName, 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['Code Smell', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35'])
        i = 0
        while i < len(correlations.S1):
            string = 'S' + str(i+1) 
            filewriter.writerow([string, correlations.S1[i], correlations.S2[i], correlations.S3[i], correlations.S4[i], correlations.S5[i], correlations.S6[i], correlations.S7[i], correlations.S8[i], correlations.S9[i], correlations.S10[i], correlations.S11[i], correlations.S12[i], correlations.S13[i], correlations.S14[i], correlations.S15[i], correlations.S16[i], correlations.S17[i], correlations.S18[i], correlations.S19[i], correlations.S20[i], correlations.S21[i], correlations.S22[i], correlations.S23[i], correlations.S24[i], correlations.S25[i], correlations.S26[i], correlations.S27[i], correlations.S28[i], correlations.S29[i], correlations.S30[i], correlations.S31[i], correlations.S32[i], correlations.S33[i], correlations.S34[i], correlations.S35[i]])
            i += 1  

def plot_corr_clustering(df,size,n):

    # Compute the correlation matrix for the received dataframe
    correlations = df.corr()
    
    i = 0
    while i < (len(correlations)):
        value = -1
        j = 0
        string = 'S' + str(i+1)
        while j < (len(correlations[string])):
            if numpy.isnan(correlations[string][j]):
                if (i+1) == int(correlations.columns[j][1]):
                    value = 1
                else:
                    value = 0    
                correlations[string][j] = value
            j += 1
        i += 1
        
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(correlations, cmap='RdYlGn')
    plt.xticks(range(len(correlations.columns)), correlations.columns, rotation=90)
    plt.yticks(range(len(correlations.columns)), correlations.columns)
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)
    plt.savefig('Results/CorrelationClustering/clustering'+str(n)+'.png')

n = 1
while n <= numFiles:
    names = ['Classes', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'SMELLS', 'HASS','CHANGES', 'HASC']
    fileName = DIR + "/" + str(n) + ".csv"
    data = pandas.read_csv(fileName, header=0, names=names)
    data.drop(['Classes', 'SMELLS', 'HASS','CHANGES', 'HASC'], axis = 1, inplace = True)  
    names = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35']

    plot_corr(data,18,n)
    
    cluster_th = 4

    X = data.corr().values
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')

    columns = [data.columns.tolist()[i] for i in list(numpy.argsort(ind))]
    df = data.reindex_axis(columns, axis=1)

    unique, counts = numpy.unique(ind, return_counts=True)
    counts = dict(zip(unique, counts))

    i = 0
    j = 0
    columns = []
    for cluster_l1 in set(sorted(ind)):
        j += counts[cluster_l1]
        sub = df[df.columns.values[i:j]]
        if counts[cluster_l1]>cluster_th:        
            X = sub.corr().values
            d = sch.distance.pdist(X)
            L = sch.linkage(d, method='complete')
            ind = sch.fcluster(L, 0.5*d.max(), 'distance')
            col = [sub.columns.tolist()[i] for i in list((numpy.argsort(ind)))]
            sub = sub.reindex_axis(col, axis=1)
        cols = sub.columns.tolist()
        columns.extend(cols)
        i = j
    df = df.reindex_axis(columns, axis=1)
    
    plot_corr_clustering(df,18,n)

    n += 1      