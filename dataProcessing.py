import matplotlib.pyplot as plt
import os, os.path
import numpy
import csv
import pandas
import scipy.cluster.hierarchy as sch

DIR = 'Datasets'
numFiles = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) - 1

n = 1
while n <= numFiles:
    names = ['Classes', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'SMELLS', 'HASS','CHANGES', 'HASC']
    fileName = DIR + "/" + str(n) + ".csv"
    data = pandas.read_csv(fileName, header=0, names=names)
    data.drop(['Classes', 'SMELLS', 'HASS','CHANGES', 'HASC'], axis = 1, inplace = True)  
    names = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35']

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
    
    size = 18 
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(correlations, cmap='RdYlGn')
    plt.xticks(range(len(correlations.columns)), correlations.columns, rotation=90)
    plt.yticks(range(len(correlations.columns)), correlations.columns)

    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)

    plt.savefig('Correlation/correlation'+str(n)+'.png')

    n += 1      