import sys, os
import argparse
import ROOT
import numpy as np
from ROOT import *
from keras import models
from train import parse_yaml, load_data
from sklearn.externals import joblib
from sklearn.preprocessing import *

def plot_response(y_pred, y_test, output, filename, eta=None, pt=None):
	"""
	Plot response 2D histogram 
	"""	
	ROOT.gROOT.SetBatch(ROOT.kTRUE) # do not print outputs of draw or load graphics
	
	print('Plotting pt error...')

	# Get arrays
	y_pred_pt = y_pred[:, 0]
	y_pred_eta = y_pred[:, 1]
	y_test_pt = y_test[:, 0]
	y_test_eta = y_test[:, 1]
	
	# Pt bins
	bins1 = 15; low1 = 20.; high1 = 100.
	pt_bins = np.linspace(low1, high1,  num=bins1)
	
	# Eta bins
	bins2 = 10; low2 = -3.0; high2 = 3.0
	eta_bins = np.linspace(low2, high2, num=bins2)

	# Define histograms
	hist_pred, _, _ = np.histogram2d(y_pred_pt, y_pred_eta, bins=[pt_bins, eta_bins], weights=y_pred_pt)
	hist_test, _, _ = np.histogram2d(y_test_pt, y_test_eta, bins=[pt_bins, eta_bins], weights=y_test_pt)
	response = np.divide(hist_pred, hist_test)

	# Plot histogram
	if eta:
		index = np.digitize(eta, eta_bins)
		response_slice = response[:, index - 1]
		response_hist = TH1F("response_hist", "", bins1, low1, high1)
	
	if pt:
		index = np.digitize(pt, pt_bins)
		response_slice = response[index - 1, :]	
		response_hist = TH1F("response_hist", "", bins2, low2, high2)
	
	len_ = len(response_slice)
	for i in range(len_):
		weight = response_slice[i]
		response_hist.AddBinContent(i, weight)
	
	cst = TCanvas("cst","cst", 10, 10, 1000, 1000) # define canvas	
	response_hist.SetStats(False)
	response_hist.Draw()	
	
	# Set title/axis labels
	if pt: 
		response_hist.SetTitle('Pta='+str(pt)+' response: '+filename)
		response_hist.GetXaxis().SetTitle('Eta')
	
	if eta: 
		response_hist.SetTitle('Eta='+str(eta)+' response: '+filename)
		response_hist.GetXaxis().SetTitle('Pt')

	leg = TLegend(0.6, 0.75, 0.9, 0.9) # create legend
	leg.SetNColumns(1)
	
	entry1 = leg.AddEntry("hist_regression","Response","l")
	entry1.SetLineWidth(2)
	entry1.SetLineColor(2)
		
	leg.Draw()
	cst.Update()
	if pt: cst.SaveAs(output +"_response_pt="+ str(pt) +".pdf")
	if eta: cst.SaveAs(output +"_response_eta="+ str(eta) +".pdf")

def plot_error(y_pred, y_test, array, bins, low, high, output, filename, 
			   plot=None, param=None, paramRange=(None,None)):
	"""
	Plot error histogram (by parameter)
	"""	
	ROOT.gROOT.SetBatch(ROOT.kTRUE) # do not print outputs of draw or load graphics

	print('Plotting error...') 
	# get indexes
	plot_parameterList = ['pt','eta','phi']
	parameterList = ['classification','jet_pt','jet_eta','jet_phi']
	plot_idx = plot_parameterList.index(plot)
	array_idx = parameterList.index('jet_'+plot)
	if param: param_idx = parameterList.index(param)

	# get errors
	regression_error = (y_pred[:, plot_idx] - y_test[:, plot_idx] / y_test[:, plot_idx])
	reference_error = (array[:, array_idx] - y_test[:, plot_idx] / y_test[:, plot_idx])

	# define histograms
	hist_regression= TH1F("regression_error", "", bins, low, high)
	hist_reference = TH1F("reference_error", "", bins, low, high)
	
	num_entries = array.shape[0]
	for i in range(num_entries):
		if param:
			if paramRange[0] < array[i][param_idx] < paramRange[1]:
				hist_regression.Fill(regression_error[i])
				hist_reference.Fill(reference_error[i])
		else:
			hist_regression.Fill(regression_error[i])
			hist_reference.Fill(reference_error[i])

	hist_regression.Sumw2() # handle errors
	hist_reference.Sumw2()
	
	cst = TCanvas("cst","cst", 10, 10, 1000, 1000) # define canvas	
	hist_regression.SetStats(False)
	
	hist_regression.SetLineWidth(2) # set width + colors
	hist_regression.SetLineColor(2)
	hist_reference.SetLineWidth(2)
	hist_reference.SetLineColor(4)

	hist_regression.Draw()	
	hist_reference.Draw('same')
	
	hist_regression.SetTitle(plot+' error: '+filename) # set title and axis labels
	hist_regression.GetXaxis().SetTitle('Mean Error')
	hist_regression.GetYaxis().SetTitle('Entries')

	leg = TLegend(0.6, 0.75, 0.9, 0.9) # create legend
	leg.SetNColumns(1)
	
	entry1 = leg.AddEntry("hist_regression","regression error","l")
	entry1.SetLineWidth(2)
	entry1.SetLineColor(2)
	
	entry2 = leg.AddEntry("hist_reference","reference error","l")
	entry2.SetLineWidth(2)
	entry2.SetLineColor(4)
	
	leg.Draw()
	cst.Update()
	if param: cst.SaveAs(output + "_" + plot + "_" + param + "_error.pdf") # save as pdf
	else: cst.SaveAs(output + "_" + plot + "_error.pdf")

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

	# Constants
	yaml_config = parse_yaml(options.config)	
	filename = yaml_config['Filename']
	output = output_plots + filename

	# Predict values		
	_, x_test, _, y_test, array = load_data(options.load)
	print('Calculating predictions...') 	
	model = models.load_model(options.directory + filename + '.h5') 	
	y_pred = model.predict(x_test)

	# Rescale values
	scaler_filename = options.load.replace('.hdf5', '_scaler.pkl')	
	scaler = joblib.load(scaler_filename) 
	y_test = scaler.inverse_transform(y_test)
	y_pred = scaler.inverse_transform(y_pred)
	
	# Plots
	plot_error(y_pred, y_test, array, 50, -100, 100, output, filename, plot='pt')
	plot_error(y_pred, y_test, array, 15, -5, 5, output, filename, plot='eta')
	plot_error(y_pred, y_test, array, 15, -5, 5, output, filename, plot='phi')
	
	plot_response(y_pred, y_test, output, filename, eta=0.5, pt=None)
	plot_response(y_pred, y_test, output, filename, eta=1.5, pt=None)
	plot_response(y_pred, y_test, output, filename, eta=2.5, pt=None)
	plot_response(y_pred, y_test, output, filename, eta=None, pt=30)
	plot_response(y_pred, y_test, output, filename, eta=None, pt=50)
	plot_response(y_pred, y_test, output, filename, eta=None, pt=90)
