import scipy
from scipy import stats
import numpy as np
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
#from matplotlib import ply
import matplotlib.pyplot as plt
from random import randint
import argparse
from numpy import genfromtxt
from numpy import add


def get_data():
    f = open('./data/20201021_developmental_transcriptome_48h.txt', 'r+')
    lines = f.readlines() # read old content
    f.seek(0) # go back to the beginning of the file
    ch = f.read(1)
    if ch != 'X':
        f.seek(0)
        f.write("X,") # write new content at the beginning
        for line in lines: # write old content after new
            f.write(line)   
    f.close()


    M1_data = genfromtxt('./data/20201021_adult_connectome_M1.txt', delimiter=',', dtype = "str", skip_header=1)
    #print (M1_data)
    #print(M1_data.shape)

    M5_data = genfromtxt('./data/20201021_adult_connectome_M5.txt', delimiter=',', dtype = "str", skip_header=1)
    #print (M5_data)
    #print(M5_data.shape)

    Trans_data = genfromtxt('./data/20201021_developmental_transcriptome_48h.txt', delimiter=',', dtype = "str", skip_header=1)
    #print (Trans_data)
    #print(Trans_data.shape)


    # extract parts of an array
    #M1_sub = M1_data[1:10, 2]
    #print(M1_sub)
    #print(M1_sub.shape)

    #M5_sub = M5_data[1:10, 2]
    #print(M5_sub)
    #print(M5_sub.shape)

    # covert extracted parts from String to Float
    #y1 = M1_sub.astype(np.float)
    #y5 = M5_sub.astype(np.float)

    # add to converted arrays
    #print (add(y1,y5))

    return M1_data, M5_data, Trans_data

def np_to_edgelist(data):

    print(data[0:10])
    neurons = np.unique(data[:,[0,1]])

    print(neurons)

    neuron_to_ind = {neurons[ind]:ind for ind in range(neurons.shape[0])}
    ind_to_neuron = {ind:neurons[ind] for ind in range(neurons.shape[0])}

    edgelist = []
    
    for i in range(data.shape[0]):
        edgelist.append([neuron_to_ind[data[i,0]], neuron_to_ind[data[i,1]], data[i,2]])

    print(edgelist[0:10])

