import h5py
import numpy as np
import pandas as pd
from utilities import progress
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
#seed = 42
#numpy.random.seed(seed)

def convert_sequence(yaml_config, parameters, labels,  parameters_df, labels_df, parameters_val, labels_val): 
	"""
	Return: Numpy array (Events, Particles, Features)
	Input: yaml_config    | Dictionary of config options
		   parameters     | List of parameters
		   labels         | List of labels
	       parameters_df  | Parameters dataframe
		   labels_df      | Labels dataframe
		   parameters_val | Parameters array
		   labels_val     | Labels array
	"""
	print('Converting data to sequence...')

	# Allocate space
	parameters_seq = np.zeros((len(labels_df), yaml_config['MaxParticles'], len(parameters) - 1))
	
	# Progress
	event_num = 0. ; total_num = len(labels_df)

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
		
		event_num += 1.
		progress.update_progress(event_num/total_num)
	
	parameters_val = parameters_seq

	return parameters_val, labels_val

def convert_image(yaml_config, parameters, labels, parameters_df, labels_df, parameters_val, labels_val):
	"""
	Return: Numpy array (Events, Eta, Phi, Features)
	Input: yaml_config    | Dictionary of config options
		   parameters     | List of parameters
		   labels         | List of labels
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

	# Progress
	event_num = 0. ; total_num = len(labels_df)

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

		event_num += 1.
		progress.update_progress(event_num/total_num)

	parameters_val = parameters_image

	return parameters_val, labels_val

def convert_vector(yaml_config, parameters, labels, parameters_df, labels_df, parameters_val, labels_val):
	"""
	Return: Numpy array (Events, Features)
	Input: yaml_config    | Dictionary of config options
		   parameters     | List of parameters
		   labels         | List of labels
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

	# Progress
	event_num = 0. ; total_num = len(labels_df)

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
			
		event_num += 1.
		progress.update_progress(event_num/total_num)
	
	parameters_val = parameters_vec

	return parameters_val, labels_val

def normalize(yaml_config, x_train, x_test):
	"""
	Return: Scaled x_train, x_test
	Input: yaml_config | Dictionary of config options
		   x_train     | Training data
		   x_test      | Testing data
	"""
	if (yaml_config['NormalizeInputs'] and yaml_config['InputType'] == 'dense'):
		print('Normalizing data...')
		
		subarrays = []
		size = x_train.shape[1]
		features = size/yaml_config['MaxParticles']
		for i in range(0, size, features):
			subarrays.append(x_train[:,i:i+features])

		scaler = StandardScaler().fit(np.concatenate(subarrays))
				
		for i in range(0, size, features):
			x_train[:,i:i+features] = scaler.transform(x_train[:,i:i+features])
			x_test[:,i:i+features] = scaler.transform(x_test[:,i:i+features])

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

def get_features(options, yaml_config):
	"""
	Return: x_train, y_train, x_test, y_test
	Input: options     | Arguments object
		   yaml_config | Dictionary of config options 
	"""	
	h5File = h5py.File(options.Input)
 	array = h5File[options.tree][()]
	print(array.shape)
	print(array.dtype.names)

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
		parameters_val, labels_val = convert_sequence(yaml_config, parameters, labels, 
						                              parameters_df, labels_df, parameters_val, labels_val)

	elif yaml_config['InputType'] == 'image':
		parameters_val, labels_val = convert_image(yaml_config, parameters, labels, 
						                           parameters_df, labels_df, parameters_val, labels_val)

	elif yaml_config['InputType'] == 'dense':
		parameters_val, labels_val = convert_vector(yaml_config, parameters, labels, 
					                                parameters_df, labels_df, parameters_val, labels_val)
	else:
		raise Exception('Invalid InputType')

	# Generate train/test split
	x_train, x_test, y_train, y_test = train_test_split(parameters_val, labels_val, test_size=0.25, random_state=42)
	x_train, x_test = normalize(yaml_config, x_train, x_test)

	return x_train, x_test, y_train, y_test


