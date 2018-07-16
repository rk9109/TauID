import sys, string
import h5py
import argparse
import numpy as np
from ROOT import *

def update_progress(progress):
	"""
	Return: Progress Bar
	Input: progress | Fraction completed from 0->1
	"""
	barLength = 20
	status = ''

	if (progress >= 1):
		progress = 1
		status = "Done...\r\n"

	block = int(round(barLength*progress))
	text = '\rPercent: [{0}] {1}% {2}'.format('#'*block + '-'*(barLength - block), progress*100, status)
	sys.stdout.write(text)
	sys.stdout.flush()

def convert_data(tree):
	"""
	docstring
	"""
	event_num = 0.
	particle_num = 0
	total_num = tree.GetEntries()

	# Parameter lists
	pt = []
	eta = []
	phi = []
	energy = []
	charge = []
	photon_ID = []; electron_ID = []; hadron_ID = []
	jet_pt = []; jet_eta = []; jet_phi = []; jet_energy = []
	classification = []

	for event in tree:
		for jet_num, jet_id in enumerate(event.genjetid):  # iterate through jet
			for k, _ in enumerate(event.genindex):         # iterate through jet particles
				if (event.genindex[k] == jet_num):

					if (abs(event.genid[k]) == 11):
						photon_ID.append(1)
						electron_ID.append(0)
						hadron_ID.append(0)

					elif (event.genid[k] == 22):
						photon_ID.append(0)
						electron_ID.append(1)
						hadron_ID.append(0)

					elif (abs(event.genid[k]) > 40):
						photon_ID.append(0)
						electron_ID.append(0)
						hadron_ID.append(1)

					else: continue
					
					# particle parameters
					pt.append(event.genpt[k])
					eta.append(event.geneta[k])
					phi.append(event.genphi[k])
					energy.append(event.genenergy[k])
					charge.append(event.gencharge[k])

					# jet parameters
					jet_pt.append(event.genjetpt[jet_num])
					jet_eta.append(event.genjeteta[jet_num])
					jet_phi.append(event.genjetphi[jet_num])
					jet_energy.append(event.genjetenergy[jet_num])
					
					if (abs(jet_id) == 15):
						classification.append(1)
					else: classification.append(0)
					
					particle_num += 1

		event_num += 1.
		update_progress(event_num/total_num)

	fields =[('pt','f8'), ('eta','f8'), ('phi','f8'), ('energy','f8'), ('charge','i4'),
			 ('photon_ID','i4'), ('electron_ID','i4'), ('hadron_ID','i4'),
			 ('jet_pt','f8'), ('jet_eta','f8'), ('jet_phi','f8'), ('jet_energy','f8'),
			 ('classification', 'i4')]

	data = np.zeros(particle_num, dtype=fields)

	data['pt'] = pt
	data['eta'] = eta
	data['phi'] = phi
	data['energy'] = energy
	data['charge'] = charge
	data['photon_ID'] = photon_ID; data['electron_ID'] = electron_ID; data['hadron_ID'] = hadron_ID
	data['jet_pt'] = jet_pt; data['jet_eta'] = jet_eta; data['jet_phi'] = jet_phi; data['jet_energy'] = jet_energy
	data['classification'] = classification

	return data

if __name__ == "__main__":	
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', help='ROOT filename')
	parser.add_argument('-t', '--tree', dest='tree', default='GenNtupler/gentree', help='tree name')
 	options = parser.parse_args()
	
	filename = options.filename
	print('Converting %s -> %s...'%(filename, filename.replace('.root', '.z')))

	# Convert TTree to numpy structured array
	rf = TFile(filename)              # open file
	tree = rf.Get(options.tree)       # get TTree
	arr = convert_data(tree)
	
	h5File = h5py.File(filename.replace('.root','.z'),'w')
	h5File.create_dataset(options.tree, data=arr,  compression='lzf')
	h5File.close()
	del h5File


