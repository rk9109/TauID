import sys, os
import argparse
import ROOT
from ROOT import *
from utilities import progress
from keras import models
from train import parse_yaml, load_data

def plot_efficiency(y_pred, y_true, array, parameter, bins, low, high, output, filename, cuts_dict, plot=None):
	"""
	Plot efficiency histogram
	"""
	ROOT.gROOT.SetBatch(ROOT.kTRUE) # do not print outputs of draw or load graphics
	
	# Intialize histograms
	loose = TH1F("correct-low", "1", bins, low, high)
	med = TH1F("correct-medium", "2", bins, low, high)
	tight = TH1F("correct-tight", "3", bins, low, high)
	total = TH1F("total", "4", bins, low, high)
	
	# Get indexes	
	parameter_list = ['classification','jet_pt','jet_eta','jet_phi','jet_index']
	param_idx = parameter_list.index(parameter)
	
	# Get cutoffs
	wp_tight = cuts_dict['tight'][0] 
	wp_med = cuts_dict['medium'][0]
	wp_loose = cuts_dict['loose'][0]

	# Iterate through array
	num = 0.; total_num = array.shape[0]
	
	for idx in range(total_num):
		classification = y_true[idx]
		param_val = array[idx][param_idx]
		y = y_pred[idx]
		
		if (classification == 1) and (plot == 'signal'):
			if y > wp_tight: tight.Fill(param_val)
			if y > wp_med: med.Fill(param_val)
			if y > wp_loose: loose.Fill(param_val)
			
			total.Fill(param_val)
		
		if (classification == 0) and (plot == 'background'):
			if y > wp_tight: tight.Fill(param_val)
			if y > wp_med: med.Fill(param_val)
			if y > wp_loose: loose.Fill(param_val)
			
			total.Fill(param_val)
		
		num += 1
		progress.update_progress_inline('Calculating efficiency...', num/total_num)

	cst = TCanvas("cst","cst", 10, 10, 1000, 1000) # define canvas	
	
	# Create graphs
	T1 = TGraphAsymmErrors(loose, total, "cl = 0.683 b(1,1)")
	T2 = TGraphAsymmErrors(med, total, "cl = 0.683 b(1,1)")
	T3 = TGraphAsymmErrors(tight, total, "cl = 0.683 b(1,1)")
	
	T1.SetMarkerColor(2); T1.SetMarkerStyle(8); T1.SetLineColor(2)
	T2.SetMarkerColor(3); T2.SetMarkerStyle(8); T2.SetLineColor(3)
	T3.SetMarkerColor(4); T3.SetMarkerStyle(8); T3.SetLineColor(4)
						
	mg = TMultiGraph() # create multigraph
	mg.Add(T1); mg.Add(T2); mg.Add(T3)
	
	mg.Draw('ap')
	mg.SetTitle(plot+' efficiency vs. '+parameter+': '+filename) # set title and axis labels
	mg.GetXaxis().SetTitle(str(parameter))
	mg.GetYaxis().SetTitle('efficiency')
	mg.GetYaxis().SetRangeUser(0., 1.2)
	if (plot == 'background'): mg.GetYaxis().SetRangeUser(0., 0.25)

	leg = TLegend(0.6, 0.8, 0.9, 0.9) # create legend
	leg.SetNColumns(1)
	
	entry1 = leg.AddEntry("T1","Loose cutoff","p")
	entry1.SetMarkerColor(2)
	entry1.SetMarkerStyle(8)

	entry2 = leg.AddEntry("T2","Medium cutoff","p")
	entry2.SetMarkerColor(3)
	entry2.SetMarkerStyle(8)

	entry3 = leg.AddEntry("T3","Tight cutoff","p")
	entry3.SetMarkerColor(4)
	entry3.SetMarkerStyle(8)
	
	leg.Draw()
	cst.Update()
	cst.SaveAs(output+'_'+plot+'_'+parameter+".pdf") # save as pdf
	
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
	y_pred = model.predict(x_test).ravel()
	
	# Plot Efficiency vs. parameter
	bins = 15; low = 20; high = 150;
	parameter = 'jet_pt'
	cuts_dict = {'tight': 0.75, 'medium': 0.5, 'loose': 0.25}
	
	plot_efficiency(y_pred, y_test, array, parameter, bins, low, high, output, filename, cuts_dict, plot='signal')

