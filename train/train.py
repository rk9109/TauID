import sys, os
import h5py
import yaml
import argparse
import keras
import numpy as np
import pandas as pd
from models import models
from features import get_features
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
#seed = 42
#numpy.random.seed(seed)

def save_model(model, outfile_name):
	"""
	Return: None
	Input: model    | Keras model
		   filename | Filename for output
	"""
	model_yaml = model.to_yaml()
	with open(outfile_name + '.yaml', 'w') as yaml_file:
		yaml_file.write(model_yaml)
	model_json = model.to_json()
	with open(outfile_name + '.json', 'w') as json_file:
		json_file.write(model_json)
	model.save_weights(outfile_name + '.h5')
	print('Saved model')

	return None

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
	parser.add_argument('-i', '--input', dest='Input', help='input ROOT file')
	parser.add_argument('-t', '--tree', dest='tree', default='GenNtupler/gentree', help='input ROOT tree')
	parser.add_argument('-o', '--output', dest='output', default='saved-models/', help='output directory')
	parser.add_argument('-c', '--config', dest='config', help='configuration file')
 	options = parser.parse_args()

	# Check output directory
	if not os.path.isdir(options.output):
		print('Specified directory not found. Creating new directory...')
		os.mkdir(options.output)
	
	# Train model
	filename = options.config.replace('.yml', '')

	# gen_model = getattr(models, yaml_config['KerasModel'])

	x_train, x_test, y_train, y_test = get_features(options) # generate data
	print(x_train)
	print(y_train)

	# model = gen_model(x_train.shape, 1, yaml_config['Loss'])
	# model, history, _, _ = train_model(x_train, y_train, x_test, y_test, model, 1024, 1024)

	# save_model(model, options.output + '/' + filename)
	
