import h5py
import argparse
import numpy as np
from utilities import progress
from convert import delta_phi 
from ROOT import *

def create_jets(event):
	"""
	Create jets from particle flow inputs
	Return: List of jet candidates (seed, [jet_particles])
	"""
	part_candidates = []
	for part in event.pf:
		part_candidates.append(part)
	
	# Sort particle candidates by Pt
	part_candidates_sorted = sorted(part_candidates, key=lambda x: x[0].Pt())[::-1]

	used_candidates = []
	jet_candidates = []
	for part in part_candidates_sorted:
		if len(jet_candidates) >= 5: break # Maximum 5 jets/event
		# Seed critera: Charged hadron
		#               Pt > 20
		#               Eta < 2.5
		if (part[1] == 211) and (abs(part[0].Eta() < 2.5)) and (part not in used_candidates):
			jet = []; seed = part[0]
			jet_vecSum = TLorentzVector()
			jet_vecSum.SetPtEtaPhiE(0., 0., 0., 0.)

			for cand in part_candidates:
				if (seed.DeltaR(cand[0]) < 0.4) and (cand not in used_candidates):
					jet_vecSum += cand[0]
					jet.append(cand)
					used_candidates.append(cand)

			if jet_vecSum.Pt() > 20:
				jet_candidates.append((seed, jet))
	
	return jet_candidates

def create_taus(event):
	"""
	Create taus from gen inputs
	Return: List of tau candidate 4-vectors 
	"""
	indices = []           # List of indices
	gen_candidates = []    # List of gen candidates: (cand, index)
	for idx, cand in enumerate(event.gen):
		index = event.tauIndex[idx]
		if index not in indices: indices.append(index)
		gen_candidates.append((cand, index))

	tau_candidates = []    # List of tau 4-vectors	
	for index in indices:
		tau_vecSum = TLorentzVector()
		tau_vecSum.SetPtEtaPhiE(0., 0., 0., 0.)

		hadron_decay = False
		for gen_cand in gen_candidates:
			if gen_cand[1] == index:
				tau_vecSum += gen_cand[0][0]
				if abs(gen_cand[0][1]) == 211:
					hadron_decay = True

		# Tau criteria: Hadronic decay
		#               Pt < 20
		#               Eta < 2.5
		if (hadron_decay) and (tau_vecSum.Pt() > 20) and (abs(tau_vecSum.Eta()) < 2.5):
			tau_candidates.append(tau_vecSum)
	
	return tau_candidates

def match_taus(jet_candidates, tau_candidates):
	"""
	Match reconstructed taus to jets
	Return: List of jets ([jet_particles], seed, tau)
	"""
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

	return jets

def convert_data(tree, number=None, regression=False):
	"""
	Particle flow inputs => Numpy structured array
	"""
	event_num = 0.    # Event counter 
	jet_num = 0       # Jet counter
	particle_num = 0  # Particle counter
	if number: total_num = number
	else: total_num = int(tree.GetEntries())

	# Parameter lists
	pt = []
	eta = []
	phi = []
	energy = []
	photon_ID = []; electron_ID = []; muon_ID; neutral_hadron_ID = []; charged_hadron_ID = []
	jet_pt = []; jet_eta = []; jet_phi = []; jet_index = []
	if regression: tau_pt = []; tau_eta = []; tau_phi = []
	classification = []

	for event in tree:
		if event_num == total_num: break	
		
		jet_candidates = create_jets(event)
		tau_candidates = create_taus(event)
		jets = match_taus(jet_candidates, tau_candidates)
	
		# Fill parameters lists
		for jet, seed, tau in jets:
			if regression and (not tau): continue
			for part, ID in jet:

				# ID Parameters
				if ID == 22:
					photon_ID.append(1)
					electron_ID.append(0)
					muon_ID.append(0)
					neutral_hadron_ID.append(0)
					charged_hadron_ID.append(0)

				elif abs(ID) == 11:
					photon_ID.append(0)
					electron_ID.append(1)
					muon_ID.append(0)
					neutral_hadron_ID.append(0)
					charged_hadron_ID.append(0)

				elif abs(ID) == 13:
					photon_ID.append(0)
					electron_ID.append(0)
					muon_ID.append(1)
					neutral_hadron_ID.append(0)
					charged_hadron_ID.append(0)
				
				elif ID == 130:
					photon_ID.append(0)
					electron_ID.append(0)
					muon_ID.append(0)
					neutral_hadron_ID.append(1)
					charged_hadron_ID.append(0)

				elif ID == 211:
					photon_ID.append(0)
					electron_ID.append(0)
					muon_ID.append(0)
					neutral_hadron_ID.append(0)
					charged_hadron_ID.append(1)

				else: continue
				
				# particle parameters
				eta_val = seed.Eta() - part.Eta()
				phi_val = delta_phi(seed.Phi(), part.Phi())
				pt.append(part.Pt())
				eta.append(eta_val)
				phi.append(phi_val)
				energy.append(part.E())

				# jet parameters
				jet_pt.append(seed.Pt())
				jet_eta.append(seed.Eta())
				jet_phi.append(seed.Phi())
				jet_index.append(jet_num)

				# tau parameters
				if tau:
					if regression:
						tau_pt.append(tau.Pt())
						tau_eta.append(tau.Eta())
						tau_phi.append(tau.Phi())
					classification.append(1)

				else: classification.append(0)
				
				particle_num += 1
			jet_num += 1
		event_num += 1	
		progress.update_progress(event_num/total_num)

	fields =[('pt','f8'), ('eta','f8'), ('phi','f8'), ('energy','f8'),
			 ('photon_ID','i4'), ('electron_ID','i4'), ('muon_ID','i4'), ('neutral_hadron_ID','i4'), ('charged_hadron_ID','i4'),
			 ('jet_pt','f8'), ('jet_eta','f8'), ('jet_phi','f8'), ('jet_index','i8'), ('classification','i4')]
	
	if regression: fields.extend([('tau_pt','f8'), ('tau_eta','f8'), ('tau_phi','f8')])

	data = np.zeros(particle_num, dtype=fields)

	data['pt'] = pt
	data['eta'] = eta
	data['phi'] = phi
	data['energy'] = energy
	data['photon_ID'] = photon_ID; data['electron_ID'] = electron_ID; data['muon_ID'] = muon_ID
	data['neutral_hadron_ID'] = neutral_hadron_ID; data['charged_hadron_ID'] = charged_hadron_ID
	data['jet_pt'] = jet_pt; data['jet_eta'] = jet_eta; data['jet_phi'] = jet_phi
	if regression: data['tau_pt'] = tau_pt; data['tau_eta'] = tau_eta; data['tau_phi'] = tau_phi
	data['jet_index'] = jet_index
	data['classification'] = classification
	
	return data

if __name__ == "__main__":	
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', help='ROOT filename')
	parser.add_argument('-n', '--number', dest='number', default=0, help='number of events')
	parser.add_argument('-t', '--tree', dest='tree', default='dumpP4/objects', help='tree name')
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
		arr = convert_data(tree, number=int(options.number), regression=False)
		h5File = h5py.File(filename.replace('.root','.z'),'w')
	
	if regression:
		print('Converting %s -> %s...'%(filename, filename.replace('.root', '_regression.z')))
		arr = convert_data(tree, number=int(options.number), regression=True)
		h5File = h5py.File(filename.replace('.root','_regression.z'),'w')
	
	h5File.create_dataset(options.tree, data=arr,  compression='lzf')
	h5File.close()
	del h5File

