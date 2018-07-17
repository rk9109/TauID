import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from processdata import *

def plot_history(history, name=None, save=False):
	"""
	Return Accuracy/Loss curve
	"""
	# summarize history for accuracy
	plt.figure(1)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	if (save): plt.savefig(name + '_accuracy.png')

	# summarize history for loss
	plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	if (save): plt.savefig(name + '_loss.png')
	
	return None

def plot_roc(model, x_test, y_test, x_back, y_back, name=None, save=False):
	"""
	Return ROC curve
	"""	
	x_test, y_test = remove_background(x_test, y_test)
	x_data = np.concatenate(x_test, x_back, axis=1)
	y_data = np.concatenate(y_test, y_back, axis=1)

	y_pred = model.predict(x_data).ravel()
	fpr, tpr, thresholds = roc_curve(y_data, y_pred)
	
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(tpr, fpr, label='Keras')
	plt.xlabel('Signal efficiency')
	plt.ylabel('Background efficiency')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()
	if (save): plt.savefig(name + '_roc.png')

	return fpr, tpr, thresholds

def get_workingpoints(fpr, tpr, thresholds):
	"""
	Return (Loose, Medium, Tight) working points
	'Performance of Tau-lepton reconstruction and identification in CMS' [https://arxiv.org/pdf/1109.6034.pdf]
	"""
	loose = 0.01
	medium = 0.005
	tight = 0.0025

	loose_cut, medium_cut, tight_cut = 0, 0, 0

	for idx, thr in enumerate(thresholds):
		if ((fpr[idx] > loose) and (loose_cut != 0)):
			loose_cut = thr
			print('Loose cut: ' + str(thr) + ' TPR: ' + str(fpr[idx]))
		if ((fpr[idx] > medium) and (medium_cut != 0)):
			medium_cut = thr	
			print('Medium cut: ' + str(thr) + ' TPR: ' + str(fpr[idx]))
		if ((fpr[idx] > tight) and (tight_cut != 0)):
			tight_cut = thr
			print('Tight cut: ' + str(thr) + ' TPR: ' + str(fpr[idx]))
	
	return (loose_cut, medium_cut, tight_cut)

def plot_pt_efficiency(model, x_test, y_test, wp):
	"""
	Plot Efficiency vs. Pt for working point
	"""
	return None

def plot_eta_efficiency(model, x_test, y_test):
	"""	
	Plot Eta vs. Pt for working point
	"""
	return None

	


