import pandas
import os, os.path
import re 
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.decomposition import PCA

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
                    listConnected.append("S"+str(res[0]))
                array.append(counter)       
            j += 1
        connected.append(listConnected)    
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
    txt = []
    while i < (len(col_names)-5):
        string = 'S' + str(i+1) 
        nodes.append([string, X2d[i][0], X2d[i][1], connected[i]])
        txt.append(string)
        i += 1

    # create graph    
    
    G = nx.Graph()
    i = 0
    while i < len(nodes):
        G.add_node(nodes[i][0],pos=(nodes[i][1], nodes[i][2]))
        j = 0
        listEdges = []
        while j < len(nodes[i][3]):
            listEdges.append((nodes[i][0],nodes[i][3][j]))
            j += 1 
        G.add_edges_from(listEdges)
        i += 1
        
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            opacity=0.8,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2),
        hoverinfo='text')   
        
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: ' + str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Code Smells Correlation',
                titlefont_size=18,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=40,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    
    fig.update_traces(textposition='top center')
    
    i = 0
    while i < len(nodes):
        fig.add_annotation(
            go.layout.Annotation(
                    x=nodes[i][1],
                    y=nodes[i][2],
                    text=nodes[i][0])
        )
        
        i += 1
        
    fig.update_annotations(dict(
        xref="x",
        yref="y",
        showarrow=True,
        arrowhead=False,
        ax=0,
        ay=-40
    )) 
       
    fig.update_layout(showlegend=False)
    fig.show()
    
    n += 1        
    