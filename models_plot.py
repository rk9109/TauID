import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

def plot_image(parameters):
	"""
	docstring
	"""
	plt.figure(1) 
	# plt.imshow()
	cbar = plt.colorbar()
	cbar.set_label('Pt')
	plt.xlabel('Eta')
	plt.ylabel('Pphi')
	plt.show()

	return None

def plot_history(history):
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

	# summarize history for loss
	plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	return None

def plot_roc(model, x_test, y_test):
	"""
	Return ROC curve
	"""
	y_pred = model.predict(x_test).ravel()
	fpr, tpr, thresholds = roc_curve(y_test, y_pred)

	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(tpr, fpr, label='Keras')
	plt.xlabel('Signal efficiency')
	plt.ylabel('Background efficiency')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()

	return None
