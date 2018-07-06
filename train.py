from models import *
from models_plot import *
from processdata import *
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

	test_loss, test_acc = model.evaluate(x_test, y_test, batch_size = batch)
	print('\nLoss on test set: ' + str(test_loss) + ' Accuracy on test set: ' + str(test_acc))
	
	return model, history, test_loss, test_acc

# ---- Copy Data ----

print('Copying files...')

signal_file = h5py.File('GluGluHToTauTau_Base.hdf5', 'r')
x_data = signal_file['x_data'][:] 
y_data = signal_file['y_data'][:]

# ---- Fit Data ----

model = one_layer_dense(x_data.shape[1], 15)
model, history, _, _ = train(x_data, y_data, model, 1000, 1024)

plot_history(history)
plot_roc(model, x_test, y_test)

