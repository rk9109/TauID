import math, random, h5py
import ROOT, sys, os, re, string
import numpy as np
from predict import update_progress
from ROOT import *

def create_data(filename, treeNum, name, randomize=True):
	"""
	Return: h5file
	Input: filename  | ROOT file
	       treeNum   | ROOT tree number 
		   name      | Filename
		   randomize | Randomly order particles
	"""
	data_file = h5py.File(name + '.hdf5', 'w') 
	rf = TFile(filename)      # open file
	tree = rf.Get(treeNum)    # get TTree
	
	event_num = 0.
	total_num = tree.GetEntries()
	
	x_arr = []
	y_arr = []
	
	for event in tree:
		for jet_num, jet_id in enumerate(event.genjetid):  # iterate through jets
			part_candidates = []

			for k, _ in enumerate(event.genindex):         # iterate through jet particles
				if (event.genindex[k] == jet_num):
					part_candidates.append([event.genpt[k], event.geneta[k], event.genphi[k], event.genenergy[k],
								            event.gencharge[k], event.genid[k]])
			
			part_candidates.sort(key = lambda x: x[0], reverse = True)

			while (len(part_candidates) < 20):
				part_candidates.append([0, 0, 0, 0, 0, 0])
	
			candidates = part_candidates[:20]
			if (randomize): random.shuffle(candidates)
			candidates_flat = [item for sublist in candidates for item in sublist]	
			candidates_flat.extend([event.genjetpt[jet_num], event.genjeteta[jet_num], event.genjetphi[jet_num],
					                event.genjetenergy[jet_num]])
			x_arr.append(candidates_flat)

			if (abs(jet_id) == 15): y_arr.append([1])
			else: y_arr.append([0])
	
		event_num += 1.
		update_progress(event_num/total_num)

	x_nparr = np.array(x_arr)
	y_nparr = np.array(y_arr)
	x_data = data_file.create_dataset('x_data', data = x_nparr)
	y_data = data_file.create_dataset('y_data', data = y_nparr)
	
	print('\nFile Generated!')
	
	return None

# ---------------

filename1 = "/home/drankin/TauID/GenNtuple_GluGluHToTauTau_M125_13TeV_powheg_pythia8.root"
filename2 = "/home/drankin/TauID/GenNtuple_ZprimeToTauTau_M-3000_TuneCP5_13TeV-pythia8-tauola.root"
filename3 = "/home/drankin/TauID/GenNtuple_QCD_HT300to500_TuneCP5_13TeV-madgraph-pythia8.root"
filename4 = "/home/drankin/TauID/GenNtuple_QCD_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8.root"
treeNum = "GenNtupler/gentree"

create_data(filename1, treeNum, 'GluGluHToTauTau_Shuffled')
create_data(filename2, treeNum, 'ZPrimeToTauTau_Shuffled')
#create_data(filename3, treeNum, 'QCD300To500_All')
#create_data(filename4, treeNum, 'QCD1500To2000_All')





