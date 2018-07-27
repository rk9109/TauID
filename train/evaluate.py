import sys, os
import argparse
import json
import numpy as np
from keras import models
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from train import parse_yaml, get_data

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
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(output + '_accuracy.png')

	# summarize history for loss
	print('Plotting loss...')
	plt.figure(2)
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('Model loss: ' + filename)
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(output + '_loss.png')
	
	return None

def plot_roc(model, x_data, y_data, output, filename):
	"""
	Return ROC curve
	"""	
	print('Plotting roc...')
	y_pred = model.predict(x_data).ravel()
	fpr, tpr, thresholds = roc_curve(y_data, y_pred)

	plt.figure(3)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Keras')
	plt.xlabel('Background efficiency')
	plt.ylabel('Signal efficiency')
	plt.title('ROC curve: ' + filename)
	plt.legend(loc='best')
	plt.savefig(output + '_roc.png')

	return tpr, fpr, thresholds

def get_workingpoints(tpr, fpr, thresholds, output):
	"""
	Return (Loose, Medium, Tight) working points
	'Performance of Tau-lepton reconstruction and identification in CMS' [https://arxiv.org/pdf/1109.6034.pdf]
	"""
	loose = 0.01
	medium = 0.005
	tight = 0.0025

	loose_cut, medium_cut, tight_cut = 0, 0, 0

	f = open(output + '_wp.txt', 'w+')
	
	for idx, thr in enumerate(thresholds):
		if ((fpr[idx] > loose) and (loose_cut == 0)):
			loose_cut = thr
			print('Loose cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]) + ' FPR: ' + str(fpr[idx]))
			f.write('Loose cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]) + ' FPR: ' + str(fpr[idx]) + '\n')
		if ((fpr[idx] > medium) and (medium_cut == 0)):
			medium_cut = thr	
			print('Medium cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]) + ' FPR: ' + str(fpr[idx]))
			f.write('Medium cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]) + ' FPR: ' + str(fpr[idx]) + '\n')
		if ((fpr[idx] > tight) and (tight_cut == 0)):
			tight_cut = thr
			print('Tight cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]) + ' FPR: ' + str(fpr[idx]))
			f.write('Tight cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]) + ' FPR: ' + str(fpr[idx]) + '\n')
	
	f.close()

	return (loose_cut, medium_cut, tight_cut)

def plot_efficiency(model, x_data, y_data, wp, output, filename):
	"""
	docstring
	"""
	raise Exception('Efficiency plot not implemented')

	return None

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--load', dest='load', help='load data file')
	parser.add_argument('-d', '--directory', dest='directory', help='input models directory')
	parser.add_argument('-c', '--config', dest='config', help='configuration file')
 	options = parser.parse_args()
	
	# Check output directory
	output_plots = 'saved-plots-higgs/'

	if not os.path.isdir(output_plots):
		print('Specified save directory not found. Creating new directory...')
		os.mkdir(output_plots)

	# Load model	
	yaml_config = parse_yaml(options.config)	
	filename = yaml_config['Filename']
	output = output_plots + filename

	model = models.load_model(options.directory + filename + '.h5') 	
	
	# Plot loss/accuracy history
	with open(options.directory + filename + '_history.json', 'r') as json_file:
		history = json.load(json_file)
	plot_history(history, output, filename)	

	# Plot ROC curve
	_, x_test, _, y_test = get_data(options.load)
	tpr, fpr, thresholds = plot_roc(model, x_test, y_test, output, filename)
	
	# Get working points
	loose_cut, medium_cut, tight_cut = get_workingpoints(tpr, fpr, thresholds, output)

	# TODO Plot background/efficiency vs. Pt
	


