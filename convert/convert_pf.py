import h5py
import argparse
import numpy as np
from utilities import progress
from convert import delta_phi 
from ROOT import *

def convert_data(tree, number=None):
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
	photon_ID = []; lepton_ID = []; neutral_hadron_ID = []; charged_hadron_ID = []
	jet_pt = []; jet_eta = []; jet_phi = []; jet_energy = []; jet_index = []
	tau_pt = []; tau_eta = []; tau_phi = []; tau_energy = []
	classification = []
	
	# Test Constants
	taus_reconstructed = 0
	taus_matched = 0
	
	for event in tree:
		if event_num == total_num: break
			
		part_candidates = []
		for part in event.pf:
			part_candidates.append(part)
				
		# Sort particle candidates by 'PT'
		part_candidates_sorted = sorted(part_candidates, key=lambda x: x[0].Pt())[::-1]
		
		used_candidates = []
		jet_candidates = []   # List of jet candidates: (seed, [jet_particles])
		for part in part_candidates_sorted:
			if len(jet_candidates) > 5: break # Maximum 5 jets/event
			# Seed criteria: Charged hadron
			#                Eta < 2.5
			if (part[1] == 211) and (abs(part[0].Eta() < 2.5)) and (part not in used_candidates):
				jet = []; seed = part[0]
				for cand in part_candidates:
					if (seed.DeltaR(cand[0]) < 0.4) and (cand not in used_candidates):
						used_candidates.append(cand)
						jet.append(cand)
				jet_candidates.append((seed, jet))

		indices = []          # List of indices
		gen_candidates = []   # List of gen candidates: (vec, index)
		for idx, gen in enumerate(event.gen):
			index = event.tauIndex[idx]
			if index not in indices: indices.append(index)
			gen_candidates.append((gen, index))
		
		tau_vec = []          # List of tau 4-vectors
		for index in indices:
			vec_sum = TLorentzVector()
			vec_sum.SetPtEtaPhiE(0., 0., 0., 0.)
		
			hadron_decay = False
			for gen_cand in gen_candidates:
				if gen_cand[1] == index:
					vec_sum += gen_cand[0][0]
					if abs(gen_cand[0][1]) == 211:
						hadron_decay = True
			# Tau criteria: Hadronic decay
			#               Pt > 15
			#               Eta < 2.5
			if (hadron_decay) and (vec_sum.Pt() > 15) and (abs(vec_sum.Eta()) < 2.5):
				tau_vec.append(vec_sum)
		
		# Match pf-gen candidates
		jets = []             # List of jets
		used_candidates = []
		for seed, jet in jet_candidates:
			tau = False
			for vec in tau_vec:
				if (seed.DeltaR(vec) < 0.4) and (vec not in used_candidates):
					tau = vec
					used_candidates.append(vec)
					break
			jets.append((jet, seed, tau))
		
		taus_matched += len(used_candidates)
		taus_reconstructed += len(tau_vec)
		
		# Fill parameters lists
		for jet, seed, tau in jets:
			for part_vec, ID in jet:
				
				# ID Parameters
				if ID == 22:
					photon_ID.append(1)
					lepton_ID.append(0)
					neutral_hadron_ID.append(0)
					charged_hadron_ID.append(0)

				elif (abs(ID) in [11, 13]):
					photon_ID.append(0)
					lepton_ID.append(1)
					neutral_hadron_ID.append(0)
					charged_hadron_ID.append(0)

				elif ID == 130:
					photon_ID.append(0)
					lepton_ID.append(0)
					neutral_hadron_ID.append(1)
					charged_hadron_ID.append(0)

				elif ID == 211:
					photon_ID.append(0)
					lepton_ID.append(0)
					neutral_hadron_ID.append(0)
					charged_hadron_ID.append(1)

				else: continue
				
				# particle parameters
				eta_val = seed.Eta() - part_vec.Eta()
				phi_val = delta_phi(seed.Phi(), part_vec.Phi())
				pt.append(part_vec.Pt())
				eta.append(eta_val)
				phi.append(phi_val)
				energy.append(part_vec.E())

				# jet parameters
				jet_pt.append(seed.Pt())
				jet_eta.append(seed.Eta())
				jet_phi.append(seed.Phi())
				jet_energy.append(seed.E())
				jet_index.append(jet_num)

				# tau parameters
				if tau:
					tau_pt.append(tau.Pt())
					tau_eta.append(tau.Eta())
					tau_phi.append(tau.Phi())
					tau_energy.append(tau.E())
					classification.append(1)

				else:
					tau_pt.append(0)
					tau_eta.append(0)
					tau_phi.append(0)
					tau_energy.append(0)
					classification.append(0)
				
				particle_num += 1
			jet_num += 1
		event_num += 1	
		progress.update_progress(event_num/total_num)

	fields =[('pt','f8'), ('eta','f8'), ('phi','f8'), ('energy','f8'),
			 ('photon_ID','i4'), ('lepton_ID','i4'), ('neutral_hadron_ID','i4'), ('charged_hadron_ID','i4'),
			 ('jet_pt','f8'), ('jet_eta','f8'), ('jet_phi','f8'), ('jet_energy','f8'), ('jet_index','i8'),
			 ('tau_pt','f8'), ('tau_eta','f8'), ('tau_phi','f8'), ('tau_energy','f8'), ('classification','i4')]

	data = np.zeros(particle_num, dtype=fields)

	data['pt'] = pt
	data['eta'] = eta
	data['phi'] = phi
	data['energy'] = energy
	data['photon_ID'] = photon_ID; data['lepton_ID'] = lepton_ID
	data['neutral_hadron_ID'] = neutral_hadron_ID; data['charged_hadron_ID'] = charged_hadron_ID
	data['jet_pt'] = jet_pt; data['jet_eta'] = jet_eta; data['jet_phi'] = jet_phi; data['jet_energy'] = jet_energy
	data['tau_pt'] = tau_pt; data['tau_eta'] = tau_eta; data['tau_phi'] = tau_phi; data['tau_energy'] = tau_energy
	data['jet_index'] = jet_index
	data['classification'] = classification
	
	print 'Taus matched: ', taus_matched
	print 'Total reconstructed: ', taus_reconstructed
	return data

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
	arr = convert_data(tree, int(options.number))

	h5File = h5py.File(filename.replace('.root','.z'),'w')	
	h5File.create_dataset(options.tree, data=arr,  compression='lzf')
	h5File.close()
	del h5File

