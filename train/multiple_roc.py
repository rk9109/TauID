import sys, os
import argparse
import numpy as np
from keras import models
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from train import parse_yaml, load_data
from evaluate import get_workingpoints

def plot_multiple_roc(y_preds, y_tests, arrays, labels, output, filename):
	"""
	Return multiple ROC curves
	"""
	print('Plotting multiple ROCs...')

	# Initialize plot
	plt.figure(3)
	plt.plot([0, 1], [0, 1], 'k--')
	
	# Iterate through lists
	for idx, array in enumerate(arrays):
		y_pred = y_preds[idx]
		y_test = y_tests[idx]

		# PT cuts: 20 < x < 100
		indices_low = np.where(array[1] < 20.0)[0].tolist() 
		indices_high = np.where(array[1] > 100.0)[0].tolist()
		indices = indices_low + indices_high
		y_pred = np.delete(y_pred, indices, axis=0)
		y_test = np.delete(y_test, indices, axis=0)
		
		# Plot ROC
		fpr, tpr, thresholds = roc_curve(y_test, y_pred)	
		plt.plot(tpr, fpr, label=labels[idx])
		
		# Plot working points
		cuts_dict = get_workingpoints(tpr, fpr, thresholds, output)
		plt.plot(cuts_dict['loose'][1], cuts_dict['loose'][2], 'rx')
		plt.plot(cuts_dict['medium'][1], cuts_dict['medium'][2], 'bx')
		plt.plot(cuts_dict['tight'][1], cuts_dict['tight'][2], 'gx')
	
	plt.legend()
	plt.xlabel('Signal efficiency')
	plt.ylabel('Background efficiency')
	plt.yscale('log')
	plt.axis([0, .6, 0, .1])
	plt.title('ROC curve: ' + filename)
	plt.savefig(output + '_roc.png')
	
	return None

if __name__ == "__main__":	
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--load', dest='load', help='load data file')
 	options = parser.parse_args()
	
	# Directory paths
	directory1 = 'saved-models/lstm_norm/lstm_norm.h5'
	directory2 = 'saved-models/gru_norm/gru_norm.h5'
	directory3 = 'saved-models/conv1D_norm/conv1D_norm.h5'
	labels = ['lstm', 'gru', 'conv1D']

	# Check output directory
	output_plots = 'saved-plots/'

	if not os.path.isdir(output_plots):
		print('Specified save directory not found. Creating new directory...')
		os.mkdir(output_plots)

	# Constants	
	filename = 'sequence_norm'
	output = output_plots + filename
	
	# Load data
	_, x_test1, _, y_test1, array1 = load_data(options.load)
	_, x_test2, _, y_test2, array2 = load_data(options.load)
	_, x_test3, _, y_test3, array3 = load_data(options.load)
	
	print('Calculating predictions...') 	
	model1 = models.load_model(directory1) 	
	model2 = models.load_model(directory2) 	
	model3 = models.load_model(directory3) 	
	y_pred1 = model1.predict(x_test1).ravel()
	y_pred2 = model2.predict(x_test2).ravel()
	y_pred3 = model3.predict(x_test3).ravel()
	
	# Initialize lists
	arrays = [array1, array2, array3]
	y_preds  = [y_pred1, y_pred2, y_pred3]
	y_tests  = [y_test1, y_test2, y_test3]
	
	# Plot multiple ROC curve
	plot_multiple_roc(y_preds, y_tests, arrays, labels, output, filename)
