import pandas
import os, os.path
import numpy as np
import csv
import sympy
from sklearn.decomposition import PCA
import itertools
import copy
import networkx as nx
import matplotlib.pyplot as plt
import re 

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
    connected = []
    totalValue = len(pima.S1)
    i = 0
    
    while i < (len(col_names)-5):
        counter = totalValue
        j = 0
        string = 'S' + str(i+1) 
        listConnected = []
        while j < (len(col_names)-5):
            stringNew = 'S' + str(j+1)
            if i == j:
                array.append(0)
            else:       
                if pima[stringNew][j] == 1 and pima[string][j] == 1:
                    counter -= 1  
                    temp = re.findall(r'\d+', stringNew) 
                    res = list(map(int, temp)) 
                    listConnected.append(res[0])
                array.append(counter)       
            j += 1
        connected.append(listConnected)    
        i += 1  
          
    i = 0
    lengthNumber = len(str(max(array)))
                          
    while i < len(array):
        if lengthNumber > 3:
            array[i] = array[i] / 1000.0
        else:
            if lengthNumber == 2:
                array[i] = array[i] / 100.0   
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
    
    nodes = []
    i = 0
    while i < (len(col_names)-5):
        string = 'S' + str(i+1) 
        nodes.append([string, X2d[i][0], X2d[i][1], connected[i]])
        i += 1
    
    print(nodes)
    # create graph    
    
    G = nx.Graph()
    G.add_edge(1,2)
    G.add_edge(1,3)
    nx.draw(G, with_labels=True)
    #plt.show()
    
    n += 1        
    