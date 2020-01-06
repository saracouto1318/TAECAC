import pandas as pd
import os, os.path
import csv
import sympy

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
    i = 0

    totalValue = len(pima.S1)

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
        
    i = 0
    start = 1
    end = 10
    width = end - start
    while i < len(array):
        if array[i] != 0:
            array[i] = (array[i] - min(array))/(max(array) - min(array)) * width + start
        i += 1
        
    matrix = []

    i = 0
    while i < len(array):
        matrix.append([array[i], array[i+1], array[i+2], array[i+3], array[i+4], array[i+5], array[i+6], array[i+7], array[i+8], array[i+9], array[i+10], array[i+11], array[i+12], array[i+13], array[i+14], array[i+15], array[i+16], array[i+17], array[i+18], array[i+19], array[i+20], array[i+21], array[i+22], array[i+23], array[i+24], array[i+25], array[i+26], array[i+27], array[i+28], array[i+29], array[i+30], array[i+31], array[i+32], array[i+33], array[i+34]])
        i += 35 
        
    distances = np.array(matrix)

    n = len(distances)
    X = sympy.symarray('x', (n, n - 1))

    for row in range(n):
        X[row, row:] = [0] * (n - 1 - row)

    for point2 in range(1, n):

        expressions = []

        for point1 in range(point2):
            expression = np.sum((X[point1] - X[point2]) ** 2) 
            expression -= distances[point1,point2] ** 2
            expressions.append(expression)

        X[point2,:point2] = sympy.solve(expressions, list(X[point2,:point2]))[1]
    
    newFileName = "DistancesProcessing/Smells-Distances" + str(n) + ".csv"            
    with open(newFileName, 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['Code Smell', 'X', 'Y'])
        i = 0
        while i < (len(col_names)-5):
            string = 'S' + str(i+1) 
            filewriter.writerow([string, X[i][0], X[i][1])
            i += 1
            
    # read csv
    # create graph        
    
    n += 1        