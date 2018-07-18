from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import *
from keras.callbacks import EarlyStopping
import h5py

def onelayer_model(input_dim, nclasses, loss):
	"""
	One hidden layer
	"""
	model = Sequential()
	
	layers = [Dense(input_dim=input_dim, units=25, kernel_initializer='random_uniform', activation='relu'), 
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation='sigmoid')]
	
	for layer in layers:
		model.add(layer)

	model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

	return model

def twolayer_model(input_dim, nclasses, loss):
	"""
	Two hidden layers
	"""
	model = Sequential()
	
	layers = [Dense(input_dim=input_dim, units=25, kernel_initializer='random_uniform', activation='relu'), 
			  Dense(units=10, kernel_initializer='random_uniform', activation='relu'),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation='sigmoid')]
	
	for layer in layers:
		model.add(layer)

	model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

	return model

def threelayer_model(input_dim, nclasses, loss):
	"""
	Three hidden layers
	"""
	model = Sequential()
	
	layers = [Dense(input_dim=input_dim, units=25, kernel_initializer='random_uniform', activation='relu'), 
			  Dense(units=25, kernel_initializer='random_uniform', activation='relu'),
			  Dense(units=10, kernel_initializer='random_uniform', activation='relu'),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation='sigmoid')]

	for layer in layers:
		model.add(layer)

	model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])

	return model

def conv1D_model(input_dim, nclasses, loss):
	"""
	Three convolutional layers
	"""
	model = Sequential()

	layers = [Conv1D(input_dim=input_dim, filters=8, kernel_size=8, strides=1, padding='same', 
			         kernel_initializer='he_normal', use_bias=True, activation='relu'),
			  Conv1D(filters=4, kernel_size=8, strides=2, padding='same', kernel_initializer='he_normal', 
					 use_bias=True, activation='relu'),
			  Conv1D(filters=2, kernel_size=8, strides=3, padding='same', kernel_initializer='he_normal', 
					 use_bias=True, activation='relu'),
			  Flatten(),
			  Dense(units=32, kernel_initializer='random_uniform', activation='relu'),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation='sigmoid')]
	
	for layer in layers:
		model.add(layer)
	
	model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])

	return model

def conv2D_model(input_dim, nclasses, loss): 
	"""
	Three convolutional layers
	"""
	model = Sequential()

	layers = [Conv2D(input_dim=input_dim, filters=8, kernel_size=(10,10), strides=(1,1), padding='same', 
			         kernel_initializer='he_normal', use_bias=True, activation='relu'),
			  Conv2D(filters=8, kernel_size=(10,10), strides=(1,1), padding='same', kernel_initializer='he_normal', 
					 use_bias=True, activation='relu'),
			  Conv2D(filters=8, kernel_size=(10,10), strides=(1,1), padding='same', kernel_initializer='he_normal', 
					 use_bias=True, activation='relu'),
			  Flatten(),
			  Dense(units=32, kernel_initializer='random_uniform', activation='relu'),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation='sigmoid')]
	
	for layer in layers:
		model.add(layer)
	
	model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])

	return model

def lstm_model(input_dim, nclasses, loss):
	"""
	Simple LSTM
	"""
	model = Sequential()

	layers = [LSTM(input_dim=input_dim, units=80, return_sequences=True), 
			  Flatten(),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation='sigmoid')]

	for layer in layers:
		model.add(layer)
	
	model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])

	return model

def gru_model(input_dim, nclasses, loss):
	"""
	Simple GRU
	"""
	model = Sequential()

	layers = [GRU(input_dim, units=80, return_sequences=True), 
			  Flatten(),
			  Dense(units=nclasses, kernel_initializer='random_uniform', activation='sigmoid')]

	for layer in layers:
		model.add(layer)
	
	model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])

	return model

