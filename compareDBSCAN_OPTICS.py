import numpy as np 
import pandas 
import matplotlib.pyplot as plt 
import os, os.path
from matplotlib import gridspec 
from sklearn.cluster import OPTICS, cluster_optics_dbscan 
from sklearn.preprocessing import normalize, StandardScaler

DIR = 'Processing'
numFiles = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) - 1

n = 1
while n <= numFiles:
    names = ['Code Smell', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35']
    fileName = DIR + "/Smells" + str(n) + ".csv"
    data = pandas.read_csv(fileName, header=0, names=names)
    data.drop(['Code Smell'], axis = 1, inplace = True)

    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(data) 
    
    # Normalizing the data so that the data 
    # approximately follows a Gaussian distribution 
    X_normalized = normalize(X_scaled) 
    
    # Converting the numpy array into a pandas DataFrame 
    X_normalized = pandas.DataFrame(X_normalized) 
    
    # Renaming the columns 
    X_normalized.columns = data.columns 
    
    X_normalized.head() 

    # Building the OPTICS Clustering model 
    optics_model = OPTICS(min_samples = 10, xi = 0.05, min_cluster_size = 0.05) 
    
    # Training the model 
    optics_model.fit(X_normalized)
    
    # Producing the labels according to the DBSCAN technique with eps = 2.0 
    labels2 = cluster_optics_dbscan(reachability = optics_model.reachability_, 
                                    core_distances = optics_model.core_distances_, 
                                    ordering = optics_model.ordering_, eps = 2) 
    
    # Creating a numpy array with numbers at equal spaces till 
    # the specified range 
    space = np.arange(len(X_normalized)) 
    
    # Storing the reachability distance of each point 
    reachability = optics_model.reachability_[optics_model.ordering_] 
    
    # Storing the cluster labels of each point 
    labels = optics_model.labels_[optics_model.ordering_] 

    # Defining the framework of the visualization 
    plt.figure(figsize =(10, 4)) 
    G = gridspec.GridSpec(1, 2) 
    ax2 = plt.subplot(G[0, 0]) 
    ax4 = plt.subplot(G[0, 1]) 
    
    # Plotting the OPTICS Clustering 
    colors = ['c.', 'b.', 'r.', 'y.', 'g.'] 
    for Class, colour in zip(range(0, 5), colors): 
        Xk = X_normalized[optics_model.labels_ == Class] 
        ax2.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha = 0.3) 
        
    ax2.plot(X_normalized.iloc[optics_model.labels_ == -1, 0], 
            X_normalized.iloc[optics_model.labels_ == -1, 1], 
        'k+', alpha = 0.1) 
    ax2.set_title('OPTICS Clustering') 
    
    # Plotting the DBSCAN Clustering with eps = 2.0 
    colors = ['c.', 'y.', 'm.', 'g.'] 
    for Class, colour in zip(range(0, 4), colors): 
        Xk = X_normalized.iloc[labels2 == Class] 
        ax4.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha = 0.3) 
            
    ax4.plot(X_normalized.iloc[labels2 == -1, 0], 
            X_normalized.iloc[labels2 == -1, 1], 
        'k+', alpha = 0.1) 
    ax4.set_title('DBSCAN Clustering with eps = 2.0') 
    
    
    plt.tight_layout() 
    plt.savefig('DBSCAN_OPTICS/comparison'+str(n)+'.png')
    n += 1