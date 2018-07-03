from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import *
from keras.callbacks import EarlyStopping
#from matplotlib import pyplot as plt
#from models_plot import *
from process_data import *
import h5py

def run_sequential(x_train, y_train, x_test, y_test, layers, epochs, batch, split=0.25, verbose=True):
	"""
	docstring
	"""
	model = Sequential()
	for layer in layers:
		model.add(layer)

	# Define optimization
	model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

	# Fit model
	early_stopping = EarlyStopping(monitor='val_loss', patience=10)
	history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch, 
					    callbacks=[early_stopping], validation_split=split, verbose=verbose)

	test_loss, test_acc = model.evaluate(x_test, y_test, batch_size = batch)
	print('\nLoss on test set: ' + str(test_loss) + ' Accuracy on test set: ' + str(test_acc))
	
	return model, history, test_loss, test_acc

def run_cnn(parameters):
	"""
	docstring
	"""
	return None
	
# ---- Copy Data ----

filename1 = 'GluGluHToTauTau_All'
filename2 = 'ZPrimeToTauTau_All'
filename3 = ''
filename4 = ''

print('Copying files...')

signal_file = h5py.File('ZPrimeToTauTau_Shuffled.hdf5', 'r')
x_data = signal_file['x_data'][:] 
y_data = signal_file['y_data'][:]

#x_data, y_data = scale_data(x_data, y_data)
x_data, y_data = onehot_data(x_data, y_data)
create_h5file(x_data, y_data, 'ZPrimeToTauTau_Shuffled_ScaledPtOnehot')

# ---- Process Data ----

print('Processing data...')

split = x_data.shape[0]//2

x_test, x_train = x_data[:split, :], x_data[split:, :]
y_test, y_train = y_data[:split], y_data[split:]

# ---- Training ----

print('Training...')

layers = [Dense(input_dim=164, units=25, kernel_initializer='random_uniform', activation='relu'), 
		  Dense(units=10, kernel_initializer='random_uniform', activation='relu'),
		  Dense(units=1, kernel_initializer='random_uniform', activation='sigmoid')]

model, history, _, _ = run_sequential(x_train, y_train, x_test, y_test, layers, 1000, 10, split=0.25, verbose=True)

plot_history(history)
plot_roc(model, x_test, y_test)


