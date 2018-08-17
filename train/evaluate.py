import sys, os
import argparse
import json
import numpy as np
from keras import models
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from train import parse_yaml, load_data
from efficiency import plot_efficiency

def plot_history(history, output, filename):
	"""
	Return Accuracy/Loss curve
	"""
	# summarize history for accuracy
	print('Plotting accuracy...')
	plt.figure(1)
	plt.plot(history['acc'])
	plt.plot(history['val_acc'])
	plt.title('Model accuracy:' + filename)
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='lower right')
	plt.savefig(output + '_accuracy.png')

	# summarize history for loss
	print('Plotting loss...')
	plt.figure(2)
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('Model loss: ' + filename)
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper right')
	plt.savefig(output + '_loss.png')
	
	return None

def plot_roc(y_pred, y_test, array, output, filename):
	"""
	Return ROC curve
	"""	
	print('Plotting roc...')
	
	# PT cuts: 20 < x < 100
	indices_low = np.where(array[1] < 20.0)[0].tolist() 
	indices_high = np.where(array[1] > 100.0)[0].tolist()
	indices = indices_low + indices_high
	y_pred = np.delete(y_pred, indices, axis=0)
	y_test = np.delete(y_test, indices, axis=0)
	
	fpr, tpr, thresholds = roc_curve(y_test, y_pred)
	
	# Get working points
	cuts_dict = get_workingpoints(tpr, fpr, thresholds, output)
	
	# Plot ROC
	plt.figure(3)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(tpr, 1/fpr, label='Keras')
	
	# Plot working points
	plt.plot(cuts_dict['loose'][1], 1/cuts_dict['loose'][2], 'rx')
	plt.plot(cuts_dict['medium'][1], 1/cuts_dict['medium'][2], 'bx')
	plt.plot(cuts_dict['tight'][1], 1/cuts_dict['tight'][2], 'gx')
	plt.yscale('log')

	plt.xlabel('Signal efficiency')
	plt.ylabel('1/Background efficiency')
	plt.title('ROC curve: ' + filename)
	plt.savefig(output + '_roc.png')

	return tpr, fpr, thresholds, cuts_dict

def get_workingpoints(tpr, fpr, thresholds, output):
	"""
	Return (Loose, Medium, Tight) working points
	'Performance of Tau-lepton reconstruction and identification in CMS' [https://arxiv.org/pdf/1109.6034.pdf]
	"""
	loose = 0.01
	medium = 0.005
	tight = 0.0025

	loose_cut, medium_cut, tight_cut = 0, 0, 0
	cuts_dict = {}

	f = open(output + '_wp.txt', 'w+')
	
	for idx, thr in enumerate(thresholds):
		if ((fpr[idx] > loose) and (loose_cut == 0)):
			loose_cut = thr
			cuts_dict['loose'] = (loose_cut, tpr[idx], fpr[idx])
			print('Loose cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]) + ' FPR: ' + str(fpr[idx]))
			f.write('Loose cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]) + ' FPR: ' + str(fpr[idx]) + '\n')
		
		if ((fpr[idx] > medium) and (medium_cut == 0)):
			medium_cut = thr	
			cuts_dict['medium'] = (medium_cut, tpr[idx], fpr[idx])
			print('Medium cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]) + ' FPR: ' + str(fpr[idx]))
			f.write('Medium cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]) + ' FPR: ' + str(fpr[idx]) + '\n')
		
		if ((fpr[idx] > tight) and (tight_cut == 0)):
			tight_cut = thr
			cuts_dict['tight'] = (tight_cut, tpr[idx], fpr[idx])
			print('Tight cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]) + ' FPR: ' + str(fpr[idx]))
			f.write('Tight cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]) + ' FPR: ' + str(fpr[idx]) + '\n')
	
	f.close()

	return cuts_dict

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--load', dest='load', help='load data file')
	parser.add_argument('-d', '--directory', dest='directory', help='input models directory')
	parser.add_argument('-c', '--config', dest='config', help='configuration file')
 	options = parser.parse_args()
	
	# Check output directory
	output_plots = 'saved-plots'

	if not os.path.isdir(output_plots):
		print('Specified save directory not found. Creating new directory...')
		os.mkdir(output_plots)

	# Constants	
	yaml_config = parse_yaml(options.config)	
	filename = yaml_config['Filename']
	output = output_plots + filename
	
	# Predict values 
	_, x_test, _, y_test, array = load_data(options.load)
	print('Calculating predictions...') 	
	model = models.load_model(options.directory + filename + '.h5') 	
	y_pred = model.predict(x_test).ravel()
	
	# Plot loss/accuracy history
	with open(options.directory + filename + '_history.json', 'r') as json_file:
		history = json.load(json_file)
	plot_history(history, output, filename)	

	# Plot ROC curve
	_, _, _, cuts_dict = plot_roc(y_pred, y_test, array, output, filename)
	
	# Plot Efficiency vs. parameter		
		
	# Pt plot
	parameter = 'jet_pt'
	bins = 15; low = 20; high = 125	
	plot_efficiency(y_pred,y_test,  array, parameter, bins, low, high, output, filename, cuts_dict, plot='signal')	
	bins = 15; low = 0; high = 250
	plot_efficiency(y_pred,y_test, array, parameter, bins, low, high, output, filename, cuts_dict, plot='background')

	# Eta plot
	parameter = 'jet_eta'
	bins = 15; low = -2.5; high = 2.5
	plot_efficiency(y_pred,y_test, array, parameter, bins, low, high, output, filename, cuts_dict, plot='signal')
	plot_efficiency(y_pred,y_test, array, parameter, bins, low, high, output, filename, cuts_dict, plot='background')
	
