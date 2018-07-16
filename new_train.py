import sys, os
import h5py
import yaml
import argparse
import keras
import numpy as np
import pandas as pd
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

def train_model():
	"""
	docstring
	"""

def get_features(options):
	"""
	docstring
	"""	
	h5File = h5py.File(options.Input)
 	array = h5File[options.tree][()]
	print(array.shape)
	print(array.dtype.names)

	yaml_config = parse_yaml(options.config)

	# List of parameters
	parameters = yaml_config['Inputs']
	labels = yaml_config['Labels']

	# Convert to dataframe
	parameters_df = pd.DataFrame(array, columns=parameters)
	reference_df = pd.DataFrame(array, columns=labels)
	labels_df = reference_df.drop_duplicates(subset='jet_pt', keep='first')

	if yaml_config['InputType'] == 'sequence':
		print('Converting data to sequence...')

	elif yaml_config['InputType'] == 'image':
		print('Converting data to image...')

		BinsX = yaml_config['BinsX']
		BinsY = yaml_config['BinsY']
		xbins = np.linspace(yamlConfig['MinX'], yamlConfig['MaxX'], BinsX + 1)
		ybins = np.linspace(yamlConfig['MinY'], yamlConfig['MaxY'], BinsY + 1)
		parameters.remove('Eta'); parameters.remove('Phi')

		# Allocate space
		parameters_2D = np.zeros(len(labels_df), BinsX, BinsY, len(parameters)) 	
		
		for i in range(labels_df):
			parameters_df_i = parameters_df[parameters_df['jet_pt'] == labels_df['jet_pt'].iloc(i)]
			
			eta = parameters_df_i['eta']
			phi = parameters_df_i['phi']
		
			for param_idx, param in enumerate(parameters):
				w = parameters_df_i[param] 
				hist, _, _ = np.histogram(eta, phi, weights=w, bins=(xbins, ybins))
				
				for ix in range(BinsX):
					for iy in range(BinsY):
						parameters_2D[i, ix, iy, param_idx] = hist[ix, iy]
		
		parameters_val = parameters_2D
		print('Done!')

	elif yaml_config['InputType'] == 'dense':
		print('Converting data to vector...')

	else:
		raise Exception('Invalid InputType')

	x_train, x_test, y_train, y_test = train_test_split(PARAMETERS)

	#Normalize Data

	return None

def parse_yaml(config_file):
	"""
	Return: Python dictionary
	Input: config_file | .yml configuration file
	"""
	print('Loading configuration from ' + str(config_file) + '...')
	config = open(config_file, 'r')
	return yaml.load(config)

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

	x_data, y_data = get_features(options) # DUMMY FUNCTION
	model, history, _, _ =  train_model(x_data, y_data, otherparameters) # DUMMY FUNCTION
		
	save_model(model, options.output + '/' + filename)



