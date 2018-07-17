from models import *
#from models_plot import *
from processdata import *
from new_train import *
import h5py

def train(x_data, y_data, model, epochs, batch, train_split=0.5, val_split=0.25, verbose=True):
	"""
	Train model
	"""
	# Split data
	split = int(x_data.shape[0]//(1/train_split))
	x_test, x_train = x_data[:split, :], x_data[split:, :]
	y_test, y_train = y_data[:split], y_data[split:]

	# Fit model
	early_stopping = EarlyStopping(monitor='val_loss', patience=10)
	history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch, 
					    callbacks=[early_stopping], validation_split=val_split, verbose=verbose)

	test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch)
	print('\nLoss on test set: ' + str(test_loss) + ' Accuracy on test set: ' + str(test_acc))
	
	return model, history, test_loss, test_acc

# ---- Copy Data ----

print('Copying files...')

filename1 = '/data/t3home000/rinik/GluGluHToTauTau_Base.hdf5'
filename2 = ''
filename3 = '/data/t3home000/rinik/QCD300To500_Base.hdf5'
filename4 = ''

signal_file = h5py.File(filename1, 'r')
background_file = h5py.File(filename3, 'r')
x_data = signal_file['x_data'][:] 
y_data = signal_file['y_data'][:]
#x_back = background_file['x_data'][:]
#y_back = background_file['y_data'][:]

# ---- Modify Data ----

#parameters = ['Pt', 'Energy', 'Eta', 'Phi']
#x_data, y_data = remove_parameters(x_data, y_data, parameters, 15)
#x_data, y_data = convert_image(x_data, y_data, 10, 15):
#x_data, y_data = remove_jet(x_data, y_data, 15)

# ---- Fit Data ----
model = one_layer_dense(x_data.shape[1], 1)
model, history, _, _ = train(x_data, y_data, model, 1000, 1024)

save_model(model, 'two_layer_dense') 

#plot_history(history)
#fpr, tpr, thresholds = plot_roc(model, x_data, y_data, x_back, y_back)
#loose_cut, medium_cut, tight_cut = get_workingpoints(fpr, tpr, thresholds)


