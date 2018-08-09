import sys, os
import argparse
import ROOT
from ROOT import *
from utilities import progress
from keras import models
from train import parse_yaml, load_data

def plot_efficiency(model, x_test, y_test, array, parameter, bins, low, high, output, filename, wp=0.5):
	"""
	Plot efficiency histogram
	"""
	ROOT.gROOT.SetBatch(ROOT.kTRUE) # do not print outputs of draw or load graphics
	
	print('Calculating predictions...') 
	y_pred = model.predict(x_test).ravel()
	
	# Intialize histograms
	hist1 = TH1F("correct-tau", "", bins, low, high)
	hist2 = TH1F("total-tau", "", bins, low, high)
	hist3 = TH1F("correct-other", "", bins, low, high)
	hist4 = TH1F("total-other", "", bins, low, high)
	
	# Get indexes	
	parameter_list = ['jet_pt','jet_eta','jet_phi','jet_et','jet_index','classification']
	param_idx = parameter_list.index(parameter)
	class_idx = parameter_list.index('classification')

	# Iterate through array
	num = 0.; total_num = x_test.shape[0]
	
	for idx in range(x_test.shape[0]):
		classification = array[idx][class_idx]
		param_val = array[idx][param_idx]
		y = y_pred[idx]

		if classification == 1:
			if y > wp: hist1.Fill(param_val)
			hist2.Fill(param_val)
		if classification == 0:
			if y > wp: hist3.Fill(param_val)
			hist4.Fill(param_val)
		
		num += 1
		progress.update_progress_inline('Calculating efficiency...', num/total_num)

	cst = TCanvas("cst","cst", 10, 10, 1000, 1000) # define canvas	

	# Create graphs
	T1 = TGraphAsymmErrors(hist1, hist2, "cl = 0.683 b(1,1)")
	T2 = TGraphAsymmErrors(hist3, hist4, "cl = 0.683 b(1,1)")

	T1.SetMarkerColor(4); T2.SetMarkerColor(2)
	T1.SetMarkerStyle(8); T2.SetMarkerStyle(22)
	T1.SetLineColor(4);   T2.SetLineColor(2)
					
	mg = TMultiGraph() # create multigraph
	mg.Add(T1); mg.Add(T2)
	mg.Draw('ap')
	mg.SetTitle('Efficiency vs. '+parameter+': '+filename) # set title and axis labels
	mg.GetXaxis().SetTitle(str(parameter))
	mg.GetYaxis().SetTitle('efficiency')
	mg.GetYaxis().SetRangeUser(0., 1.2)
	
	leg = TLegend(0.6, 0.8, 0.9, 0.9) # create legend
	leg.SetNColumns(1)
	
	entry1 = leg.AddEntry("T1","Signal Efficiency","p")
	entry1.SetMarkerColor(4)
	entry1.SetMarkerStyle(8)
	
	entry2 = leg.AddEntry("T2","Background Efficiency","p")
	entry2.SetMarkerColor(2)
	entry2.SetMarkerStyle(22)
	
	leg.Draw()
	cst.Update()
	cst.SaveAs(output+parameter+".pdf") # save as pdf
	
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
	
	# Values
	bins = 25; low = 0; high = 500;
	parameter = 'jet_pt'

	# Plot Efficiency vs. parameter	
	_, x_test, _, y_test, array = load_data(options.load)
	plot_efficiency(model, x_test, y_test, array, parameter, bins, low, high, output, filename, wp=0.5)
