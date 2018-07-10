import math, random, h5py
import ROOT, sys, os, re, string
import numpy as np
from predict import update_progress
from ROOT import *

def create_data(filename, treeNum, particles, name, randomize=True):
	"""
	Return: h5file
	Input: filename  | ROOT file
	       treeNum   | ROOT tree number
		   particles | Number of particles to include
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
					if (abs(event.genid[k]) == 11):
						part_candidates.append([event.genpt[k], event.geneta[k], event.genphi[k], event.genenergy[k],
								            	event.gencharge[k], 1, 0, 0])
					if (event.genid[k] == 22):
						part_candidates.append([event.genpt[k], event.geneta[k], event.genphi[k], event.genenergy[k], 
												event.gencharge[k], 0, 1, 0])
					if (abs(event.genid[k]) > 40):
						part_candidates.append([event.genpt[k], event.geneta[k], event.genphi[k], event.genenergy[k], 
												event.gencharge[k], 0, 0, 1])
			
			part_candidates.sort(key = lambda x: x[0], reverse = True)

			while (len(part_candidates) < particles):
				part_candidates.append([0, 0, 0, 0, 0, 0, 0, 0])
	
			candidates = part_candidates[:particles]
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
	print('\nFile Generated!')
	
	return None

def create_data_decay(filename, treeNum, particles, name, randomize=True):	
	"""
	Return: h5file
	Input: filename  | ROOT file
	       treeNum   | ROOT tree number
		   particles | Number of particles to include
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
		for jet_num, jet_id in enumerate(event.genjetid):   # iterate through jets
			part_candidates = []
			
			if (abs(event.genjetid[jet_num]) == 15):        # only consider tau jets
				for k, _ in enumerate(event.genindex):      # iterate through jet particles
					if (event.genindex[k] == jet_num):
						if (abs(event.genid[k]) == 11):
							part_candidates.append([[event.genpt[k], event.geneta[k], event.genphi[k], event.genet[k],
									           		 1, 0, 0], 0])
						if (event.genid[k] == 22):
							part_candidates.append([[event.genpt[k], event.geneta[k], event.genphi[k], event.genet[k], 
											   		 0, 1, 0], 0])
						if (abs(event.genid[k]) > 40):
							part_candidates.append([[event.genpt[k], event.geneta[k], event.genphi[k], event.genet[k], 
							               		     0, 0, 1], 0])
						
						if ((event.genstatus[k] == 1) and ((event.genid[k] in [11, -11, 22]) or (abs(event.genid[k]) > 40))): 
							index = k
							while ((event.genparent[index] != -2) and (abs(event.genid[index] != 15))):
								index = event.genparent[index]

							if (abs(event.genid[index]) == 15):
								part_candidates[-1][1] = 1
 						
				part_candidates.sort(key = lambda x: x[0][0], reverse = True)

				while (len(part_candidates) < particles):
					part_candidates.append([[0, 0, 0, 0, 0, 0, 0], 0])
	
				candidates = part_candidates[:particles]
				if (randomize): random.shuffle(candidates)
						
				cand_arr = []
				decay_arr = []

				for pair in candidates:
					cand_arr.append(pair[0])
					decay_arr.append(pair[1])
	
				candidates_flat = [item for sublist in cand_arr for item in sublist]	
				candidates_flat.extend([event.genjetpt[jet_num], event.genjeteta[jet_num], event.genjetphi[jet_num],
					                	event.genjetet[jet_num]])
						
				x_arr.append(candidates_flat)
				y_arr.append(decay_arr)

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
filename5 = "/home/pharris/GenNtuple.root"
treeNum = "GenNtupler/gentree"

#create_data(filename1, treeNum, 15, 'GluGluHToTauTau_Base')
#create_data(filename2, treeNum, 15, 'ZPrimeToTauTau_Base')
create_data(filename3, treeNum, 15, 'QCD300To500_Base')
create_data(filename4, treeNum, 15, 'QCD1500To2000_Base')
#create_data_decay(filename5, treeNum, 15, 'Test_Decay')


