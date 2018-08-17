import h5py
import argparse
import numpy as np
from utilities import progress
from convert import delta_phi
from convertPF import create_jets, create_taus
from ROOT import *

def initialize_hist(bins, low, high):
	"""
	Initialize histograms to fill
	"""
	hist1 = TH1F("", "", bins, low, high)
	hist2 = TH1F("", "", bins, low, high)

	return hist1, hist2

def plot_hist(hist1, hist2, title, x_label, y_label):
	"""
	Plot efficiency histograms
	"""
	cst = TCanvas("cst","cst", 10, 10, 1000, 1000)	

	# Create graphs
	T = TGraphAsymmErrors(hist1, hist2, "cl = 0.683 b(1,1)")
	T.SetMarkerColor(4); T.SetMarkerStyle(8); T.SetLineColor(4)
					
	T.Draw('ap')
	T.SetTitle(title) 
	T.GetXaxis().SetTitle(x_label)
	T.GetYaxis().SetTitle(y_label)
	T.GetYaxis().SetRangeUser(0., 1.2)
	
	cst.Update()
	cst.SaveAs(title + '.pdf')
	
	return None

def plot_data(tree, number=None):
	"""
	Plot matching efficiencies
	"""
	event_num = 0.    # Event counter 
	if number: total_num = number
	else: total_num = int(tree.GetEntries())

	taus_matched = 0
	taus_reconstructed = 0

	# Initialize histograms
	hist_pt1, hist_pt2 = initialize_hist(25, 0, 500) 
	hist_eta1, hist_eta2 = initialize_hist(15, -2.5, 2.5)	
	hist_lpt1, hist_lpt2 = initialize_hist(25, 0, 500)
	hist_leta1, hist_leta2 = initialize_hist(15, -2.5, 2.5)
	hist_slpt1, hist_slpt2 = initialize_hist(25, 0, 500)
	hist_sleta1, hist_sleta2 = initialize_hist(15, -2.5, 2.5)

	cnt = Counter()
	
	for event in tree:
		if event_num == total_num: break	
		jet_candidates = create_jets(event)
		tau_candidates = create_taus(event)

		jets = []              # List of jets
		used_candidates = []
		for seed, jet in jet_candidates:
			tau = None
			for vec in tau_candidates: 
				if (seed.DeltaR(vec) < 0.4) and (vec not in used_candidates):
					tau = vec
					used_candidates.append(vec)
					break
			jets.append((jet, seed, tau))
		
		# Fill histograms
		taus_matched += len(used_candidates)
		taus_reconstructed += len(tau_candidates)
		
		for vec in used_candidates:
			hist_pt1.Fill(vec.Pt())
			hist_eta1.Fill(vec.Eta())
		for vec in tau_candidates:
			hist_pt2.Fill(vec.Pt())
			hist_eta2.Fill(vec.Eta())

		tau_candidates = sorted(tau_candidates, key=lambda x: x.Pt())[::-1]

		if len(tau_candidates) >= 1:
			hist_lpt2.Fill(tau_candidates[0].Pt())
			hist_leta2.Fill(tau_candidates[0].Eta())
			
			if tau_candidates[0] in used_candidates:
				hist_lpt1.Fill(tau_candidates[0].Pt())
				hist_leta1.Fill(tau_candidates[0].Eta())
			
			if len(tau_candidates) >= 2:
				hist_slpt2.Fill(tau_candidates[1].Pt())
				hist_sleta2.Fill(tau_candidates[1].Eta())
				
				if tau_candidates[1] in used_candidates:
					hist_slpt1.Fill(tau_candidates[1].Pt())
					hist_sleta1.Fill(tau_candidates[1].Eta())

		event_num += 1	
		progress.update_progress(event_num/total_num)

	# Plot histograms
	print 'Taus matched: ', taus_matched
	print 'Taus reconstructed: ', taus_reconstructed
	
	plot_hist(hist_pt1, hist_pt2, 'matching_efficiency_pt', 'pt', 'efficiency')
	plot_hist(hist_lpt1, hist_lpt2, 'matching_efficiency_leadingpt', 'pt', 'efficiency')
	plot_hist(hist_slpt1, hist_slpt2, 'matching_efficiency_subleadingpt', 'pt', 'efficiency')
	plot_hist(hist_eta1, hist_eta2, 'matching_efficiency_eta', 'eta', 'efficiency')
	plot_hist(hist_leta1, hist_leta2, 'matching_efficiency_leadingeta', 'eta', 'efficiency')
	plot_hist(hist_sleta1, hist_sleta2, 'matching_efficiency_subleadingeta', 'eta', 'efficiency')

	return None

if __name__ == "__main__":	
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', help='ROOT filename')
	parser.add_argument('-n', '--number', dest='number', default=0, help='number of events')
	parser.add_argument('-t', '--tree', dest='tree', default='dumpP4/objects', help='tree name')
 	options = parser.parse_args()
	
	filename = options.filename
	print('Converting %s -> %s...'%(filename, filename.replace('.root', '.z')))

	# Convert TTree to numpy structured array	
	rf = TFile(filename)              # open file
	tree = rf.Get(options.tree)       # get TTree
	arr = plot_data(tree, number=int(options.number))
