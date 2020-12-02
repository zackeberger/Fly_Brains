from data_loading import *
import json

edges, features, _ = get_network()

# as requested in comment
exDict = {'exDict': 1}

num_nodes = len(features)
with open('M1-id_map.json', 'w') as file:
     file.write(json.dumps(exDict)) 
