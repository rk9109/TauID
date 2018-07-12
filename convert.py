import h5py
import argparse
import numpy as np
from root_numpy import root2array, tree2array
from root import *

if __name__ == "__main__":	
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', help='ROOT filename')
	parser.add_argument('-t', '--tree', dest='tree', default='GenNtupler/gentree', help='tree name')
	parser.add_argument('-s', '--seed', type=int, dest='seed', default=None, help='np.random seed')
 	options = parser.parse_args()

	print('Convering %s -> %s...'%(filename, filename.replace('.root', '.z')))

	# Convert TTree to numpy structured array
	rf = TFile(options.filename)      # open file
	tree = rf.Get(options.tree)       # get TTree
	arr = tree2array(tree)

	h5File = h5py.File(fileName.replace('.root','.z'),'w')        
    h5File.create_dataset(options.tree, data=arr,  compression='lzf')
	h5File.close()
	del h5File


