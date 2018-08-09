from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.regularizers import l1
import h5py

def onelayer_model(input_dim, nclasses, loss, output, l1Reg=0):
	"""
	One hidden layer
	"""
	model = Sequential()
	
	layers = [Dense(input_dim=input_dim, units=25, kernel_initializer='random_uniform', activation='relu',
			  W_regularizer = l1(l1Reg)), 
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation=output,
		      W_regularizer = l1(l1Reg))]
	
	for layer in layers:
		model.add(layer)

	model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

	return model

def twolayer_model(input_dim, nclasses, loss, output, l1Reg=0):
	"""
	Two hidden layers
	"""
	model = Sequential()
	
	layers = [Dense(input_dim=input_dim, units=25, kernel_initializer='random_uniform', activation='relu',
			  W_regularizer = l1(l1Reg)), 
			  Dense(units=10, kernel_initializer='random_uniform', activation='relu', 
			  W_regularizer = l1(l1Reg)),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation=output,
			  W_regularizer = l1(l1Reg))]
	
	for layer in layers:
		model.add(layer)

	model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

	return model

def threelayer_model(input_dim, nclasses, loss, output, l1Reg=0):
	"""
	Three hidden layers
	"""
	model = Sequential()
	
	layers = [Dense(input_dim=input_dim, units=25, kernel_initializer='random_uniform', activation='relu',
			  W_regularizer = l1(l1Reg)), 
			  Dense(units=25, kernel_initializer='random_uniform', activation='relu',
			  W_regularizer = l1(l1Reg)),
			  Dense(units=10, kernel_initializer='random_uniform', activation='relu',
			  W_regularizer = l1(l1Reg)),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation=output,
			  W_regularizer = l1(l1Reg))]

	for layer in layers:
		model.add(layer)

	model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])

	return model

def conv1d_model(input_shape, nclasses, loss, output, l1Reg=0):
	"""
	Three convolutional layers
	"""
	model = Sequential()

	layers = [Conv1D(input_shape=input_shape, filters=8, kernel_size=8, strides=1, padding='same', 
			         kernel_initializer='he_normal', use_bias=True, activation='relu', W_regularizer = l1(l1Reg)),
			  Conv1D(filters=4, kernel_size=8, strides=2, padding='same', kernel_initializer='he_normal', 
					 use_bias=True, activation='relu', W_regularizer = l1(l1Reg)),
			  Conv1D(filters=2, kernel_size=8, strides=3, padding='same', kernel_initializer='he_normal', 
					 use_bias=True, activation='relu', W_regularizer = l1(l1Reg)),
			  Flatten(),
			  Dense(units=32, kernel_initializer='random_uniform', activation='relu', W_regularizer = l1(l1Reg)),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation=output,
					W_regularizer = l1(l1Reg))]
	
	for layer in layers:
		model.add(layer)
	
	model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])

	return model

def conv2d_model(input_shape, nclasses, loss, output, l1Reg=0): 
	"""
	Three convolutional layers
	"""
	model = Sequential()

	layers = [Conv2D(input_shape=input_shape, filters=8, kernel_size=(10,10), strides=(1,1), padding='same', 
			         kernel_initializer='he_normal', use_bias=True, activation='relu', W_regularizer = l1(l1Reg)),
			  Conv2D(filters=8, kernel_size=(10,10), strides=(1,1), padding='same', kernel_initializer='he_normal', 
					 use_bias=True, activation='relu', W_regularizer = l1(l1Reg)),
			  Conv2D(filters=8, kernel_size=(10,10), strides=(1,1), padding='same', kernel_initializer='he_normal', 
					 use_bias=True, activation='relu', W_regularizer = l1(l1Reg)),
			  Flatten(),
			  Dense(units=32, kernel_initializer='random_uniform', activation='relu', W_regularizer = l1(l1Reg)),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation=output,
					W_regularizer = l1(l1Reg))]
	
	for layer in layers:
		model.add(layer)
	
	model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])

	return model

def lstm_model(input_shape, nclasses, loss, output, l1Reg=0):
	"""
	Simple LSTM
	"""
	model = Sequential()

	layers = [LSTM(input_shape=input_shape, units=64, return_sequences=True, W_regularizer = l1(l1Reg)),
			  LSTM(units=32, return_sequences=True, W_regularizer = l1(l1Reg)),
			  LSTM(units=16, return_sequences=True, W_regularizer = l1(l1Reg)),
			  Flatten(),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation=output, W_regularizer = l1(l1Reg))]

	for layer in layers:
		model.add(layer)
	
	model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])

	return model

def gru_model(input_shape, nclasses, loss, output, l1Reg=0):
	"""
	Simple GRU
	"""
	model = Sequential()

	layers = [GRU(input_shape=input_shape, units=64, return_sequences=True, W_regularizer = l1(l1Reg)),
			  GRU(units=32, return_sequences=True, W_regularizer = l1(l1Reg)),
			  GRU(units=16, return_sequences=True, W_regularizer = l1(l1Reg)),
			  Flatten(),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation=output,
				    W_regularizer = l1(l1Reg))]

	for layer in layers:
		model.add(layer)
	
	model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])

	return model


