import h5py
import yaml
import numpy as np
import pandas as pd
from utilities import progress

def parse_yaml(config_file):
	"""
	Return: Python dictionary
	Input: config_file | .yml configuration file
	"""
	print('Loading configuration from ' + str(config_file) + '...')
	config = open(config_file, 'r')
	return yaml.load(config)

def convert_sequence(yaml_config, parameters_df, labels_df, parameters_val, labels_val): 
	"""
	Return: Numpy array (Events, Particles, Features)
	Input: yaml_config    | Dictionary of config options
	       parameters_df  | Parameters dataframe
		   labels_df      | Labels dataframe
		   parameters_val | Parameters array
		   labels_val     | Labels array
	"""
	print('Converting data to sequence...')

	# Allocate space
	parameters_seq = np.zeros((len(labels_df), yaml_config['MaxParticles'], len(parameters) - 1))

	for i in range(len(labels_df)):
		parameters_df_i = parameters_df[parameters_df['jet_pt'] == labels_df['jet_pt'].iloc[i]]
		index_values = parameters_df_i.index.values
		parameters_val_i = parameters_val[index_values, :-1]
	
		num_particles = len(parameters_val_i)
		max_particles = yaml_config['MaxParticles']

		if num_particles > max_particles:
			parameters_val_i = parameters_val_i[0:max_particles, :]
	
		else:
			parameters_val_i = np.concatenate([parameters_val_i, np.zeros((max_particles - num_particles,
															len(parameters) - 1))])
	
		if yaml_config['Shuffle']: np.random.shuffle(parameters_val_i)
		parameters_seq[i, :, :] = parameters_val_i

	parameters_val = parameters_seq
	print('Done!')

	return parameters_val, labels_val

def convert_image(yaml_config, parameters_df, labels_df, parameters_val, labels_val):
	"""
	Return: Numpy array (Events, Eta, Phi, Features)
	Input: yaml_config    | Dictionary of config options
	       parameters_df  | Parameters dataframe
		   labels_df      | Labels dataframe
		   parameters_val | Parameters array
		   labels_val     | Labels array
	"""	
	print('Converting data to image...')

	BinsX = yaml_config['BinsX']
	BinsY = yaml_config['BinsY']
	xbins = np.linspace(yaml_config['MinX'], yaml_config['MaxX'], BinsX + 1)
	ybins = np.linspace(yaml_config['MinY'], yaml_config['MaxY'], BinsY + 1)
	parameters.remove('eta'); parameters.remove('phi')
	parameters.remove('jet_eta'); parameters.remove('jet_phi'); parameters.remove('jet_pt')
		
	# Allocate space
	parameters_image = np.zeros((len(labels_df), BinsX, BinsY, len(parameters))) 	
		
	for i in range(len(labels_df)):
		parameters_df_i = parameters_df[parameters_df['jet_pt'] == labels_df['jet_pt'].iloc[i]]
			
		eta = np.asarray(parameters_df_i['eta'])
		phi = np.asarray(parameters_df_i['phi'])
		scaled_eta = np.asarray(parameters_df_i['jet_eta']) - eta 
		scaled_phi = np.asarray(parameters_df_i['jet_phi']) - phi
		
		for param_idx, param in enumerate(parameters):
			w = np.asarray(parameters_df_i[param])
			hist, _, _ = np.histogram2d(scaled_eta, scaled_phi, weights=w, bins=(xbins, ybins))
				
			for ix in range(BinsX):
				for iy in range(BinsY):
					parameters_image[i, ix, iy, param_idx] = hist[ix, iy]
		
	parameters_val = parameters_image
	print('Done!')

	return parameters_val, labels_val

def convert_vector(yaml_config, parameters_df, labels_df, parameters_val, labels_val):
	"""
	Return: Numpy array (Events, Features)
	Input: yaml_config    | Dictionary of config options
	       parameters_df  | Parameters dataframe
		   labels_df      | Labels dataframe
		   parameters_val | Parameters array
		   labels_val     | Labels array
	"""	
	print('Converting data to vector...')
	
	max_particles = yaml_config['MaxParticles']
	vec_length = max_particles * (len(parameters) - 1)

	# Allocate space
	parameters_vec = np.zeros((len(labels_df), vec_length))

	for i in range(len(labels_df)):
		parameters_df_i = parameters_df[parameters_df['jet_pt'] == labels_df['jet_pt'].iloc[i]]
		index_values = parameters_df_i.index.values
		parameters_val_i = parameters_val[index_values, :-1]
		num_particles = len(parameters_val_i)

		if num_particles > max_particles:
			parameters_val_i = parameters_val_i[0:max_particles, :]
			
		else:
			parameters_val_i = np.concatenate([parameters_val_i, np.zeros((max_particles - num_particles,
						                      							   len(parameters) - 1))])
			
		if yaml_config['Shuffle']: np.random.shuffle(parameters_val_i)

		vector = []
		for idx in range(len(parameters_val_i)):
			vector.extend(parameters_val_i[idx])

		parameters_vec[i,:] = vector

	parameters_val = parameters_image
	print('Done!')

	return parameters_val, labels_val

def normalize(yaml_config, x_train, x_test)
	"""
	Return: Scaled x_train, x_test
	Input: yaml_config | Dictionary of config options
		   x_train     | Training data
		   x_test      | Testing data
	"""
	if (yaml_config['NormalizeInputs'] and yaml_config['InputType'] == 'dense'):
		print('Normalizing data...')
	
		# SCALE THIS DATA PROPERLY

		scaler = StandardScaler().fit(x_train)
		x_train = scaler.transform(x_train)
		x_test = scaler.transform(x_test)

	if (yaml_config['NormalizeInputs'] and yaml_config['InputType'] == 'sequence'):
		print('Normalizing data...')
		
		x_train_reshape = x_train.reshape(x_train.shape[0]*x_train.shape[1], x_train.shape[2])
		scaler = StandardScaler().fit(x_train)
		
		for part in range(x_train.shape[1]):
			x_train[:, part, :] = scaler.transform(x_train[:, part, :])
			x_test[:, part, :] = scaler.transform(x_test[:, part, :])

	if (yaml_config['NormalizeInputs'] and yaml_config['InputType'] == 'image'):
		print('Normalizing data...')
		
		raise Exception('Conv2D normalization not yet implemented')

	return x_train, x_test

def get_features(options):
	"""
	Return: x_train, y_train, x_test, y_test
	Input: options | Arguments object
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

	# Convert to numpy array
	parameters_val = parameters_df.values
	labels_val = labels_df.values[:, :-1]

	if yaml_config['InputType'] == 'sequence':
		parameters_val, labels_val = convert_sequence(yaml_config, parameters_df, labels_df, parameters_val, labels_val)

	elif yaml_config['InputType'] == 'image':
		parameters_val, labels_val = convert_image(yaml_config, parameters_df, labels_df, parameters_val, labels_val)

	elif yaml_config['InputType'] == 'dense':
		parameters_val, labels_val = convert_vector(yaml_config, parameters_df, labels_df, parameters_val, labels_val)

	else:
		raise Exception('Invalid InputType')

	# Generate train/test split
	x_train, x_test, y_train, y_test = train_test_split(parameters_val, labels_val, test_size=0.25, random_state=42)
	x_train, x_test = normalize(yaml_config, x_train, x_test)

	return x_train, x_test, y_train, y_test


