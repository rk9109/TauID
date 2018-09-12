import h5py
import argparse
import numpy as np
from utilities import progress
from ROOT import *

def delta_phi(phi1, phi2):
	pi = np.pi
	x = phi1 - phi2
	while x >= pi:
		x -= (2*pi)
	while x < -pi:
		x += (2*pi)
	return x
	
def convert_data(tree, number=None):
	"""
	Convert data for binary classification
	"""
	event_num = 0.
	jet_num = 0
	particle_num = 0
	if number: total_num = number
	else: total_num = int(tree.GetEntries())

	# Parameter lists
	pt = []
	eta = []
	phi = []
	et = []
	photon_ID = []; electron_ID = []; hadron_ID = []
	jet_pt = []; jet_eta = []; jet_phi = []; jet_et = []; jet_index = []
	classification = []

	for event in tree:
		if event_num == total_num: break
		for jet_idx, jet_id in enumerate(event.genjetid):  # iterate through jet
			for k, _ in enumerate(event.genindex):         # iterate through jet particles
				if (event.genindex[k] == jet_idx):

					if (event.genid[k] == 22):
						photon_ID.append(1)
						electron_ID.append(0)
						hadron_ID.append(0)

					elif (abs(event.genid[k]) == 11):
						photon_ID.append(0)
						electron_ID.append(1)
						hadron_ID.append(0)

					elif (abs(event.genid[k]) > 40):
						photon_ID.append(0)
						electron_ID.append(0)
						hadron_ID.append(1)

					else: continue
					
					# particle parameters
					eta_val = event.genjeteta[jet_idx] - event.geneta[k]
					phi_val = delta_phi(event.genjetphi[jet_idx], event.genphi[k])
					pt.append(event.genpt[k])
					eta.append(eta_val)
					phi.append(phi_val)
					et.append(event.genet[k])

					# jet parameters
					jet_pt.append(event.genjetpt[jet_idx])
					jet_eta.append(event.genjeteta[jet_idx])
					jet_phi.append(event.genjetphi[jet_idx])
					jet_et.append(event.genjetet[jet_idx])
					jet_index.append(jet_num)

					if (abs(jet_id) >= 4):
						classification.append(1)
					else: classification.append(0)
					
					particle_num += 1

			jet_num += 1

		event_num += 1.
		progress.update_progress(event_num/total_num)

	fields =[('pt','f8'), ('eta','f8'), ('phi','f8'), ('et','f8'),
			 ('photon_ID','i4'), ('electron_ID','i4'), ('hadron_ID','i4'),
			 ('jet_pt','f8'), ('jet_eta','f8'), ('jet_phi','f8'), ('jet_et','f8'), ('jet_index','i8'),
			 ('classification','i4')]

	data = np.zeros(particle_num, dtype=fields)

	data['pt'] = pt
	data['eta'] = eta
	data['phi'] = phi
	data['et'] = et
	data['photon_ID'] = photon_ID; data['electron_ID'] = electron_ID; data['hadron_ID'] = hadron_ID
	data['jet_pt'] = jet_pt; data['jet_eta'] = jet_eta; data['jet_phi'] = jet_phi; data['jet_et'] = jet_et
	data['jet_index'] = jet_index
	data['classification'] = classification

	return data

def convert_regression_data(tree, number=None):
	"""
	Convert data for regression
	"""
	event_num = 0.
	jet_num = 0
	particle_num = 0
	if number: total_num = number
	else: total_num = int(tree.GetEntries())

	# Parameter lists
	pt = []
	eta = []
	phi = []
	et = []
	photon_ID = []; electron_ID = []; hadron_ID = []
	jet_pt = []; jet_eta = []; jet_phi = []; jet_et = []; jet_index = []
	tau_pt = []; tau_eta = []; tau_phi = []; tau_energy = []

	for event in tree:
		if event_num == total_num: break
		for jet_idx, jet_id in enumerate(event.genjetid):  # iterate through jet
			if (abs(jet_id) >= 4):                         # consider tau jets	
				particle_vec = []
				
				for k, _ in enumerate(event.genindex):     # iterate through jet particles
					if (event.genindex[k] == jet_idx):
						if (event.genstatus[k] == 1) and ((event.genid[k] in [11, -11, 22]) 
										                   or (abs(event.genid[k]) > 40)): 
							index = k
							while ((event.genparent[index] != -2) and (abs(event.genid[index] != 15))):
								index = event.genparent[index]

							if (abs(event.genid[index]) == 15):
								vec_pt = event.genpt[k]
								vec_eta = event.geneta[k]
								vec_phi = event.genphi[k]
								vec_energy = event.genet[k]*np.cosh(vec_eta) 
								
								vec = TLorentzVector()
								vec.SetPtEtaPhiE(vec_pt, vec_eta, vec_phi, vec_energy)
								particle_vec.append(vec)
				
				if particle_vec:
					vec_sum = TLorentzVector()
					vec_sum.SetPtEtaPhiE(0., 0., 0., 0.)
					
					for vec in particle_vec:
						vec_sum += vec

					tau_pt_val = vec_sum.Pt(); tau_energy_val = vec_sum.E()
					tau_eta_val = vec_sum.Eta(); tau_phi_val = vec_sum.Phi()

					for k, _ in enumerate(event.genindex):     # iterate through jet particles
						if (event.genindex[k] == jet_idx):

							if (event.genid[k] == 22):
								photon_ID.append(1)
								electron_ID.append(0)
								hadron_ID.append(0)

							elif (abs(event.genid[k]) == 11):
								photon_ID.append(0)
								electron_ID.append(1)
								hadron_ID.append(0)

							elif (abs(event.genid[k]) > 40):
								photon_ID.append(0)
								electron_ID.append(0)
								hadron_ID.append(1)

							else: continue
					
							# particle parameters
							eta_val = event.genjeteta[jet_idx] - event.geneta[k]
							phi_val = delta_phi(event.genjetphi[jet_idx], event.genphi[k])
							pt.append(event.genpt[k])
							eta.append(eta_val)
							phi.append(phi_val)
							et.append(event.genet[k])

							# jet parameters
							jet_pt.append(event.genjetpt[jet_idx])
							jet_eta.append(event.genjeteta[jet_idx])
							jet_phi.append(event.genjetphi[jet_idx])
							jet_et.append(event.genjetet[jet_idx])
							jet_index.append(jet_num)

							# tau parameters
							tau_pt.append(tau_pt_val)
							tau_eta.append(tau_eta_val)
							tau_phi.append(tau_phi_val)
							tau_energy.append(tau_energy_val)

							particle_num += 1
					
			jet_num += 1

		event_num += 1.
		progress.update_progress(event_num/total_num)

	fields =[('pt','f8'), ('eta','f8'), ('phi','f8'), ('et','f8'),
			 ('photon_ID','i4'), ('electron_ID','i4'), ('hadron_ID','i4'),
			 ('jet_pt','f8'), ('jet_eta','f8'), ('jet_phi','f8'), ('jet_et','f8'), ('jet_index', 'i8'),
			 ('tau_pt','f8'), ('tau_eta','f8'), ('tau_phi','f8'), ('tau_energy','f8')]

	data = np.zeros(particle_num, dtype=fields)

	data['pt'] = pt
	data['eta'] = eta
	data['phi'] = phi
	data['et'] = et
	data['photon_ID'] = photon_ID; data['electron_ID'] = electron_ID; data['hadron_ID'] = hadron_ID
	data['jet_pt'] = jet_pt; data['jet_eta'] = jet_eta; data['jet_phi'] = jet_phi; data['jet_et'] = jet_et
	data['tau_pt'] = tau_pt; data['tau_eta'] = tau_eta; data['tau_phi'] = tau_phi; data['tau_energy'] = tau_energy
	data['jet_index'] = jet_index

	return data

if __name__ == "__main__":	
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', help='ROOT filename')
	parser.add_argument('-n', '--number', dest='number', default=0, help='number of events')
	parser.add_argument('-t', '--tree', dest='tree', default='GenNtupler/gentree', help='tree name')
 	options = parser.parse_args()

	# Settings
	classification = True
	regression = True

	# Get ROOT TTree	
	filename = options.filename
	rf = TFile(filename)              # open file
	tree = rf.Get(options.tree)       # get TTree
	
	# Convert TTree to numpy structured array 
	if classification:
		print('Converting %s -> %s...'%(filename, filename.replace('.root', '.z')))
		arr = convert_data(tree, int(options.number))
		h5File = h5py.File(filename.replace('.root','.z'),'w')
		h5File.create_dataset(options.tree, data=arr,  compression='lzf')
		h5File.close()
		del h5File
	
	if regression:
		print('Converting %s -> %s...'%(filename, filename.replace('.root', '_regression.z')))
		arr = convert_regression_data(tree, int(options.number))
		h5File = h5py.File(filename.replace('.root','_regression.z'),'w')	
		h5File.create_dataset(options.tree, data=arr,  compression='lzf')
		h5File.close()
		del h5File
