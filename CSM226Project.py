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


my_path = '/Users/sashaschtein/Downloads/'

f = open(my_path+'20201021_developmental_transcriptome_48h.txt', 'r+')
lines = f.readlines() # read old content
f.seek(0) # go back to the beginning of the file
ch = f.read(1)
if ch != 'X':
    f.seek(0)
    f.write("X,") # write new content at the beginning
    for line in lines: # write old content after new
        f.write(line)   
f.close()


M1_data = genfromtxt(my_path+'20201021_adult_connectome_M1.txt', delimiter=',', dtype = "str")
print (M1_data)
print(M1_data.shape)

M5_data = genfromtxt(my_path+'20201021_adult_connectome_M5.txt', delimiter=',', dtype = "str")
print (M5_data)
print(M5_data.shape)

Trans_data = genfromtxt(my_path+'20201021_developmental_transcriptome_48h.txt', delimiter=',', dtype = "str")
print (Trans_data)
print(Trans_data.shape)


# extract parts of an array
M1_sub = M1_data[1:10, 2]
print(M1_sub)
print(M1_sub.shape)

M5_sub = M5_data[1:10, 2]
print(M5_sub)
print(M5_sub.shape)

# covert extracted parts from String to Float
y1 = M1_sub.astype(np.float)
y5 = M5_sub.astype(np.float)

# add to converted arrays
print (add(y1,y5))