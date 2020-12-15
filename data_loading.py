import numpy as np


def get_data(time=48):
"""
Import the datasets
"""
    M1_data = np.genfromtxt('./data/20201021_adult_connectome_M1.txt', delimiter=',', dtype = "str", skip_header=1)
    M5_data = np.genfromtxt('./data/20201021_adult_connectome_M5.txt', delimiter=',', dtype = "str", skip_header=1)
    Trans_data = np.genfromtxt('./data/20201021_developmental_transcriptome_' + str(time) + 'h.txt', delimiter=',', dtype = "str", skip_header=1)

    return M1_data, M5_data, Trans_data


def np_to_edgelist(edges, features):
"""
Create an edgelist for the connectome network
"""
    neuron_to_feature = {features[i,0]:features[i,1:] for i in range(features.shape[0])}

    i = 0
    while(1):
        if(i >= edges.shape[0]):
            break
        if((edges[i][0] not in neuron_to_feature.keys()) or (edges[i][1] not in neuron_to_feature.keys())):
            edges = np.delete(edges, i, axis=0)
        else:
            i += 1

    #print(edges[0:10])
    neurons = np.unique(edges[:,[0,1]])

    #print(neurons)
    neuron_to_ind = {neurons[ind]:ind for ind in range(neurons.shape[0])}
    ind_to_neuron = {ind:neurons[ind] for ind in range(neurons.shape[0])}

    edgelist = []
    
    for i in range(edges.shape[0]):
        edgelist.append([neuron_to_ind[edges[i,0]], neuron_to_ind[edges[i,1]], edges[i,2]])

    neuron_to_feature = {features[i,0]:features[i,1:] for i in range(features.shape[0]) if features[i,0] in neurons}

    features_arr = np.zeros((len(neurons), features.shape[1]-1))

    for i in range(neurons.shape[0]):
        features_arr[i] = neuron_to_feature[ind_to_neuron[i]]

    return edgelist, features_arr, ind_to_neuron 


def get_network(time=48):
"""
Get network with corresponding features
"""
    M1, M5, trans_data = get_data(time=time)
    edges, features, ind_to_neurons = np_to_edgelist(M1, trans_data)

    return edges, features, ind_to_neurons


def network_to_mat(edges, features):
    e, m = features.shape
    n = len(edges)

    X = np.zeros((n, 2*m))
    Y = np.zeros(n)

    for i in range(n):
        u, v, w = edges[i]
        X[i,:m] = features[u]
        X[i,m:] = features[v]
        Y[i] = w

    return X, Y
