import sys, os
import argparse
import json
import numpy as np
from keras import models
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from train import parse_yaml, get_data

def plot_history(history, name):
	"""
	Return Accuracy/Loss curve
	"""
	# summarize history for accuracy
	plt.figure(1)
	plt.plot(history['acc'])
	plt.plot(history['val_acc'])
	plt.title('Model accuracy:' + name)
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(name + '_accuracy.png')

	# summarize history for loss
	plt.figure(2)
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('Model loss: ' + name)
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(name + '_loss.png')
	
	return None

def include_background(x_test, y_test, x_back, y_back):
	"""
	Remove background
	"""
	indices = np.where(y_test == 0)[0]
	x_test = np.delete(x_test, indices, axis=0)
	y_test = np.delete(y_test, indices, axis=0)

	x_data = np.concatenate((x_test, x_back), axis=0)
	y_data = np.concatenate((y_test, y_back), axis=0)
	
	return x_data, y_data

def plot_roc(model, x_test, y_test, x_back, y_back, name):
	"""
	Return ROC curve
	"""	
	x_data, y_data = remove_background(x_test, y_test, x_back, y_back)

	y_pred = model.predict(x_data).ravel()
	fpr, tpr, thresholds = roc_curve(y_data, y_pred)
	
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Keras')
	plt.xlabel('Background efficiency')
	plt.ylabel('Signal efficiency')
	plt.title('ROC curve: ' + name)
	plt.legend(loc='best')
	plt.savefig(name + '_roc.png')

	return tpr, fpr, thresholds

def get_workingpoints(tpr, fpr, thresholds):
	"""
	Return (Loose, Medium, Tight) working points
	'Performance of Tau-lepton reconstruction and identification in CMS' [https://arxiv.org/pdf/1109.6034.pdf]
	"""
	loose = 0.01
	medium = 0.005
	tight = 0.0025

	loose_cut, medium_cut, tight_cut = 0, 0, 0

	for idx, thr in enumerate(thresholds):
		if ((fpr[idx] > loose) and (loose_cut == 0)):
			loose_cut = thr
			print('Loose cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]))
		if ((fpr[idx] > medium) and (medium_cut == 0)):
			medium_cut = thr	
			print('Medium cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]))
		if ((fpr[idx] > tight) and (tight_cut == 0)):
			tight_cut = thr
			print('Tight cut: ' + str(thr) + ' TPR: ' + str(tpr[idx]))
	
	return (loose_cut, medium_cut, tight_cut)

def plot_efficiency(model, x_test, y_test, wp):
	"""
	Plot Efficiency vs. Pt/Eta for working point
	"""
	raise Exception('Efficiency plot not implemented')
	
	x_data, y_data = remove_background(x_test, y_test, x_back, y_back)
	
	for idx, jet in enumerate(x_data):
		prob = model.predict(jet)
		if (prob > wp and y_data[idx] == 1): signal[Pt] += 1
		if (prob < wp and y_data[idx] == 0): background[Pt] += 1
		total[Pt] += 1
	
	hist_signal, bins_signal = np.hist(parameters)
	hist_back, bins_back = np.hist(parameters)
	hist_total, bins_total = np.hist(parameters)

	# DIVIDE AND PLOT
			
	return None

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--signal', dest='signal', help='input hdf5 signal file')
	parser.add_argument('-b', '--background', dest='background', help='input hdf5 backgronud file')
	parser.add_argument('-m', '--model', dest='model', help='input saved keras model')
	parser.add_argument('-l', '--history', dest='history', help='input saved history')
	parser.add_argument('-o', '--output', dest='output', default='saved-plots/', help='output directory')
	parser.add_argument('-c', '--config', dest='config', help='configuration file')
 	options = parser.parse_args()
	
	# Check output directory
	if not os.path.isdir(options.output):
		print('Specified save directory not found. Creating new directory...')
		os.mkdir(options.output)

	model = models.load_model(options.model) 	
	yaml_config = parse_yaml(options.config)	
	filename = options.output + yaml_config['Filename']
	
	# Plot loss/accuracy history
	if (options.history):
		with open(options.history, 'r') as json_file:
			history = json.load(json_file)
		plot_history(history, filename)	

	# Plot ROC curve
	_, x_test, _, y_test = get_data(options.signal)
	_, x_back, _, y_back = get_data(options.background)
	tpr, fpr, thresholds = plot_roc(model, x_test, y_test, x_back, y_back, filename)
	
	# Get working points
	loose_cut, medium_cut, tight_cut = get_workingpoints(tpr, fpr, thresholds)

	# TODO Plot background/signal efficiency vs. Pt
	



