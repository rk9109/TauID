import sys, os
import argparse
import yaml
import json
import h5py
import keras
import numpy as np
from keras.callbacks import *
from models import models
from features import get_features
#seed = 42
#numpy.random.seed(seed)

def parse_yaml(config_file):
	"""
	Return: Python dictionary
	Input: config_file | .yml configuration file
	"""
	print('Loading configuration from ' + str(config_file) + '...')
	config = open(config_file, 'r')
	return yaml.load(config)

def save_model(model, history, output, filename):
	"""
	Return: None
	Input: model    | Keras model
		   history  | Loss history
		   filename | Filename for output
	"""
	os.mkdir(output + filename)
	outfile_name = output + filename + '/' + filename

	model_yaml = model.to_yaml()
	with open(outfile_name + '.yaml', 'w') as yaml_file:
		yaml_file.write(model_yaml)
	model_json = model.to_json()
	with open(outfile_name + '.json', 'w') as json_file:
		json_file.write(model_json)
	with open(outfile_name + '_history.json', 'w') as json_history_file:
		history_dict = history.history
		json.dump(history_dict, json_history_file)
	
	model.save_weights(outfile_name + '_weights.h5')
	model.save(outfile_name + '.h5')
	print('Saved model')

	return None

def save_data(filename, x_train, x_test, y_train, y_test):
	"""
	Save generated data
	"""
	print('Saving data...')
	h5File = h5py.File(filename + '.hdf5','w')
	h5File.create_dataset('x_train', data=x_train,  compression='lzf')
	h5File.create_dataset('x_test', data=x_test, compression='lzf')	
	h5File.create_dataset('y_train', data=y_train,  compression='lzf')
	h5File.create_dataset('y_test', data=y_test, compression='lzf')
	h5File.close()
	del h5File

	return None

def get_data(filename):
	"""
	Reload generated data
	"""
	print('Reloading data...')
	h5File = h5py.File(filename)
 	x_train = h5File['x_train']; x_test = h5File['x_test']
	y_train = h5File['y_train']; y_test = h5File['y_test']
	
	return x_train, x_test, y_train, y_test

def train_model(x_train, y_train, x_test, y_test, model, epochs, batch, val_split=0.25, verbose=True):
	"""
	Train model
	"""
	# Fit model
	early_stopping = EarlyStopping(monitor='val_loss', patience=100)
	history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch, 
					    callbacks=[early_stopping], validation_split=val_split, verbose=verbose)

	test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch)
	print('\nLoss on test set: ' + str(test_loss) + ' Accuracy on test set: ' + str(test_acc))
	
	return model, history, test_loss, test_acc

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--signal', dest='signal', help='input ROOT signal file')
	parser.add_argument('-b', '--back', dest='background', help='input ROOT background file')
	parser.add_argument('-t', '--tree', dest='tree', default='GenNtupler/gentree', help='input ROOT tree')
	parser.add_argument('-l', '--load', dest='load', help='load data file')
	parser.add_argument('-c', '--config', dest='config', help='configuration file')
	options = parser.parse_args()

	# Check output directory
	output_models = 'saved-models/'
	output_data = 'saved-data/'
	
	if not os.path.isdir(output_models):
		print('Specified output directory not found. Creating new directory...')
		os.mkdir(output_models)
	
	if not os.path.isdir(output_data):
		print('Specified save directory not found. Creating new dirctory...')
		os.mkdir(output_data)

	# Generate data
	yaml_config = parse_yaml(options.config)	
	filename = yaml_config['Filename']

	if (options.load):
		x_train, x_test, y_train, y_test = get_data(options.load)
	
	elif (options.config):
		x_train, x_test, y_train, y_test = get_features(options, yaml_config)
		save_data(output_data + filename, x_train, x_test, y_train, y_test)

	else:
		raise Exception('Load/Config file not specified.')

	# Train model
	gen_model = getattr(models, yaml_config['KerasModel'])
	
	if yaml_config['InputType'] == 'dense':
		model = gen_model(x_train.shape[1], y_train.shape[1], yaml_config['KerasLoss'])
	
	if yaml_config['InputType'] == 'image':
		model = gen_model(x_train.shape[1:], y_train.shape[1], yaml_config['KerasLoss'])
	
	if yaml_config['InputType'] == 'sequence':
		model = gen_model(x_train.shape[1:], y_train.shape[1], yaml_config['KerasLoss'])

	model, history, _, _ = train_model(x_train, y_train, x_test, y_test, model, 1024, 1024)
	
	save_model(model, history, output_models, filename)	
