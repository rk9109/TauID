import sys, os
import argparse
import h5py
import numpy as np
import pandas as pd
import ROOT
from ROOT import *
from rootpy.plotting import *

def plot_spectrum(array, parameter, low, high, bins, name):
	"""
	Plot histogram of parameter
	"""
	# Create dataframes
	parameters = pd.DataFrame(array, columns=[parameter])
	parameters = parameters.drop_duplicates(keep='first')
	labels = pd.DataFrame(array, columns=['classification', parameter])
	labels = labels.drop_duplicates(subset=parameter, keep='first')

	# Create arrays	
	parameters_val = parameters.values
	labels_val = labels.values[:, :-1]
	
	signal_indices = np.where(labels_val == 1)[0]
	background_indices = np.where(labels_val == 0)[0]
	signal_array = np.delete(parameters_val, background_indices, axis = 0)
	background_array = np.delete(parameters_val, signal_indices, axis=0)

	# Plot histogram
	for array in [signal_array, background_array]:
		plt.hist(array, bins=bins, range=(low, high))
		plt.xlabel(parameter)
		plt.ylabel('Number of events')
		plt.title(name + ': ' + parameter)
		plt.show()

	return None

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', help='ROOT filename')
	parser.add_argument('-t', '--tree', dest='tree', default='GenNtupler/gentree', help='tree name')
 	options = parser.parse_args()

	filename = options.filename	
	array_file = h5py.File(filename)
 	array = array_file[options.tree][()]
	
	# Plot spectrum
	parameter = 'jet_pt'
	plot_spectrum(array, parameter, 0, 1500, 125, 'Parameter') 

