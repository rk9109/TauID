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
"""

def one_layer_dense_binary(input_dim):
	"""
	docstring
	"""
	model = Sequential()
	
	layers = [BinaryDense(64, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='fc1', input_shape=(16,)),
			  BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn1'),
			  Activation(binary_tanh, name='act{}'.format(1)),
			  BinaryDense(32, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='fc2'),
			  BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn2'),
			  Activation(binary_tanh, name='act{}'.format(2)),
			  BinaryDense(1, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='output'),
			  BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn')]
	
	for layer in layers:
		model.add(layer)

	model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
	
	return model                                                       
"""


