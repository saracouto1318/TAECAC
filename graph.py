import pandas as pd
import csv

col_names = ['Classes', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'SMELLS', 'HASS','CHANGES', 'HASC']
# load dataset
pima = pd.read_csv("Datasets/3.3.csv", header=0, names=col_names)
pima.drop(['Classes', 'SMELLS', 'HASS','CHANGES', 'HASC'], axis = 1, inplace = True) 
pima.head()

array = []
i = 0

totalValue = len(pima.S1)

while i < (len(col_names)-5):
    counter = 0
    j = 0
    string = 'S' + str(i+1) 
    while j < (len(col_names)-5):
        stringNew = 'S' + str(j+1)
        if i == j:
            array.append(1)
        else:       
            if pima[stringNew][j] == 1 and pima[string][j] == 1:
                counter += 1  
            array.append(counter)   
        j += 1
    i += 1   

with open('Smells2.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Code Smell', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35'])
    line = 0
    i = 0
    while line < (len(col_names)-5):
        string = 'S' + str(line+1) 
        filewriter.writerow([string, array[i], array[i+1], array[i+2], array[i+3], array[i+4], array[i+5], array[i+6], array[i+7], array[i+8], array[i+9], array[i+10], array[i+11], array[i+12], array[i+13], array[i+14], array[i+15], array[i+16], array[i+17], array[i+18], array[i+19], array[i+20], array[i+21], array[i+22], array[i+23], array[i+24], array[i+25], array[i+26], array[i+27], array[i+28], array[i+29], array[i+30], array[i+31], array[i+32], array[i+33], array[i+34]])
        i += 35
        line += 1
 
col_names = ['Smells','S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35']        
pima = pd.read_csv("Smells2.csv", header=0, names=col_names)
pima.head()       
print(pima)