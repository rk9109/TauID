import numpy as np
import h5py

def create_h5file(x_data, y_data, name):
	"""
	Create h5file from data
	"""
	data_file = h5py.File(name + '.hdf5', 'w')
	data_file.create_dataset('x_data', data=x_data)
	data_file.create_dataset('y_data', data=y_data)
	print('\nFile Generated!')
	
	return None

def remove_background(x_data, y_data):
	"""
	Remove non-tau jets
	"""
	indices = np.where(y_data == 0)
	x_data = np.delete(x_data, indices, axis=0)
	y_data = np.delete(y_data, indices, axis=0)

	return x_data, y_data

def remove_jet(x_data, y_data, particles):
	"""
	Remove jet information
	"""
	indices = [0, 1, 2, 3]
	for idx, _ in enumerate(indices): indices[idx] += 8*particles
	x_data = np.delete(x_data, indices, axis=1)

	return x_data, y_data

def remove_parameters(x_data, y_data, parameters, particles):
	"""
	Remove parameter information: (Pt, Eta, Phi, Energy, Charge, ID)
	"""
	param_dict = {'Pt':[0], 'Eta':[1], 'Phi':[2], 'Energy':[3], 'Charge':[4], 'ID':[5,6,7]}
	ref_indices = []; indices = []

	for p in parameters:
		ref_indices.extend(param_dict[p])
		
	for i in range(particles):
		for idx in ref_indices:
			indices.append(idx + i*8)
	
	x_data = np.delete(x_data, indices, axis=1)

	return x_data, y_data

# ---------- Functions to be updated ----------

def scale_ptenergy(x_data, y_data):
	"""
	Scale Pt/Energy distribution
	"""
	return x_data, y_data

def convert_sequence(x_data, y_data, particles):
	"""
	Convert data to sequence
	"""
	return x_data, y_data

def convert_image(x_data, y_data, bins, particles):
	"""
	Convert data to image 
	"""
	eta_bins = np.linspace(-3, 3, bins)
	phi_bins = np.linspace(-3, 3, bins)
	
	x_cnn_data = np.zeros((x_data.shape[0], bins, bins))
	y_cnn_data = y_data

	for idx, jet in enumerate(x_data):
		for i in range(0, 8*particles, 8):
			x = np.digitize(jet[i + 1], eta_bins)
			y = np.digitize(jet[i + 2], phi_bins)
			pt = jet[i]
			x_cnn_data[idx, x - 1, y - 1] = pt

	return x_cnn_data, y_cnn_data




