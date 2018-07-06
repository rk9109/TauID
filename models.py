from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import *
from keras.callbacks import EarlyStopping
import h5py

def one_layer_dense(input_dim, nclasses):
	"""
	docstring
	"""
	model = Sequential()
	
	layers = [Dense(input_dim=input_dim, units=25, kernel_initializer='random_uniform', activation='relu'), 
			  Dense(units=10, kernel_initializer='random_uniform', activation='relu'),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation='sigmoid')]
	
	for layer in layers:
		model.add(layer)

	model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

	return model

def two_layer_dense(input_dim, nclasses):
	"""
	docstring
	"""
	model = Sequential()
	
	layers = [Dense(input_dim=input_dim, units=25, kernel_initializer='random_uniform', activation='relu'), 
			  Dense(units=10, kernel_initializer='random_uniform', activation='relu'),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation='sigmoid')]

	for layer in layers:
		model.add(layer)

	model.compile(loss ='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

	return model

def conv1D(): # ADD PARAMETERS
	"""
	Three convolutional layers
	"""
	model = Sequential()

	layers = []

	return model

def conv2D(): # ADD PARAMETERS
	"""
	Three convolutional layers
	"""
	model = Sequential()

	layers = []

	return model

def lstm(): # ADD PARAMETERS
	"""
	docstring
	"""
	model = Sequential()

	layers = []

	return model



