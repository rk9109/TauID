import numpy as np
import h5py

def scale_data(x_data, y_data):
	"""
	Scale Pt/Energy to unity
	"""
	print('Scaling data...')
	
	x_ref = np.ones_like(x_data)

	for i in range(0, 120, 6):
		x_ref[:, i] = x_data[:, 120]
		x_ref[:, i+3] = x_data[:, 123]
		
	x_ref[:, 120] = x_data[:, 120]
	x_ref[:, 123] = x_data[:, 123]
	
	x_data = x_data/x_ref
	
	print('Scaling complete!')

	return x_data, y_data

def onehot_data(x_data, y_data):
	"""
	Convert particle ID to one-hot: (photon, electron, hadron)
	"""
	print('Converting to onehot...')

	onehot = np.zeros((x_data.shape[0], 60))
	x_data = np.concatenate((x_data, onehot), axis=1)

	for idx, jet in enumerate(x_data):
		for i in range(0, 120, 6):
			if x_data[idx, i + 5] > 40: x_data[idx, i/2 + 126] = 1
			
			elif x_data[idx, i + 5] in [11, -11]: x_data[idx, i/2 + 125] = 1
			
			elif x_data[idx, i + 5] == 22: x_data[idx, i/2 + 124] = 1
	
	indices = []
	for i in range(0, 120, 6):
		indices.append(i + 5)
	
	x_data = np.delete(x_data, indices, axis=1)
	
	print('Conversion complete!')

	return x_data, y_data

def convert_image(x_data, y_data, bins):
	"""
	Convert data to image 
	"""
	eta_bins = np.linspace(-3, 3, bins)
	phi_bins = np.linspace(-3, 3, bins)
	
	x_cnn_data = np.zeros((x_data.shape[0], bins, bins))
	y_cnn_data = y_data

	for idx, jet in enumerate(x_data):
		for i in range(0, 70, 7):
			x = np.digitize(jet[i + 1], eta_bins)
			y = np.digitize(jet[i + 2], phi_bins)
			pt = jet[i]
			x_cnn_data[idx, x - 1, y - 1] = pt

	return x_cnn_data, y_cnn_data

def create_h5file(x_data, y_data, name):
	"""
	Create h5file from data
	"""
	data_file = h5py.File(name + '.hdf5', 'w')
	data_file.create_dataset('x_data', data=x_data)
	data_file.create_dataset('y_data', data=y_data)
	print('\nFile Generated!')
	
	return None




