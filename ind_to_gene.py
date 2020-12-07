import numpy as np

Trans_data = np.genfromtxt('./data/20201021_developmental_transcriptome_' + str(48) + 'h.txt', delimiter=',', dtype = "str", skip_header=0)

genes = Trans_data[0]

print(genes[3874])
print(genes[3654])
print(genes[2350])
print(genes[1225])
print(genes[927])

print(genes[2238])
print(genes[575])
print(genes[1073])
print(genes[2896])
print(genes[4021])
