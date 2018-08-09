import sys, os
import argparse
import ROOT
from ROOT import *
from keras import models
from train import parse_yaml, load_data

def plot_error(model, x_test, y_test, parameter, output, filename):
	"""
	Plot error histogram
	"""
	print('Plotting error...') 
	y_pred = model.predict(x_test).ravel()

	# Values
	bins = 25; low = 0; high = 500

	hist = TH1F("error", "", bins, low, high)
	error = (y_test - y_pred)/y_pred
	hist.Fill(error)

	cst = TCanvas("cst","cst", 10, 10, 1000, 1000) # define canvas	
	hist.SetLineWidth(2)
	hist.SetLineColor(1)
	hist.Draw()

	cst.Update()
	cst.SaveAs(filename + ".png") # save as png
	
	return None

def plot_error_parameter(model, x_test, y_test, array, parameter, output, filename):
	"""
	docstring
	"""
	return None

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--load', dest='load', help='load data file')
	parser.add_argument('-d', '--directory', dest='directory', help='input models directory')
	parser.add_argument('-c', '--config', dest='config', help='configuration file')
 	options = parser.parse_args()
	
	# Check output directory
	output_plots = 'saved-plots/'

	if not os.path.isdir(output_plots):
		print('Specified save directory not found. Creating new directory...')
		os.mkdir(output_plots)

	# Load model	
	yaml_config = parse_yaml(options.config)	
	filename = yaml_config['Filename']
	output = output_plots + filename

	model = models.load_model(options.directory + filename + '.h5') 	
	
	# Plot Error
	plot_error(model, x_test, y_test, parameter, output, filename)
