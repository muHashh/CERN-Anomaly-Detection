import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt


def mask_training_cuts_jet_based(constituents, jets):
    ''' get mask for training cuts requiring a jet-pt > 200'''
    jetPt_cut = 200
    idx_j1Pt = 10
    mask_j1 = jets[:, idx_j1Pt] > jetPt_cut
    return mask_j1


def mask_events(events, indices, values):
    mask = np.ones(len(events), dtype=bool)
    for (idx, val) in zip(indices, values):
        passed = np.abs(events[:, :, idx]) < val
        mask *= passed.all(axis=1)
    return events[mask]


def constituents_to_input_samples(constituents, mask_j1):  # -> np.ndarray
    const_j1 = constituents[:, :, :][mask_j1]
    np.random.shuffle(const_j1)
    return const_j1


def jets_to_input_samples(constituents, jets):
    mask_j1 = mask_training_cuts_jet_based(constituents, jets)
    return constituents_to_input_samples(constituents, mask_j1)


def std_norm(data, idx):
    return (data[:, :, idx] - np.mean(data[:, :, idx]))/np.std(data[:, :, idx])


def min_max_norm(data, idx):
    return (data[:, :, idx] - np.mean(data[:, :, idx]))/np.std(data[:, :, idx])


def normalize_features_jet_based(particles, feats):
    for idx in range(len(feats)):
        if not 'pt' in feats[idx]:
            # standard norm
            particles[:, :, idx] = (
                particles[:, :, idx] - np.mean(particles[:, :, idx]))/np.std(particles[:, :, idx])
        else:
            # min-max normalize pt
            particles[:, :, idx] = (particles[:, :, idx] - np.min(particles[:, :, idx])) / (
                np.max(particles[:, :, idx])-np.min(particles[:, :, idx]))
    return particles

def normalize_features_event_based(particles):
    idx_px, idx_py, idx_pz, idx_pt, idx_eta, idx_phi = range(6)
    # min-max normalize pt
    particles[:, :, idx_pt] = min_max_norm(particles, idx_pt)
    # standard normalize angles and cartesians
    for idx in (idx_px, idx_py, idx_pz, idx_eta, idx_phi):
        particles[:, :, idx] = std_norm(particles, idx)
    return particles

def normalize_features(particles):
    
    def transform_min_max(x):
        return (x-np.min(x))/(np.max(x)-np.min(x))

    def transform_mean_std(x):
        return (x-np.mean(x))/(3*np.std(x))

    idx_eta, idx_phi, idx_pt = range(3)
    # min-max normalize pt
    particles[:,:,idx_pt] = transform_min_max(particles[:,:,idx_pt])
    # standard normalize angles
    particles[:,:,idx_eta] = transform_mean_std(particles[:,:,idx_eta])
    particles[:,:,idx_phi] = transform_mean_std(particles[:,:,idx_phi])
    return particles


def normalized_adjacency(A):
    D = np.array(np.sum(A, axis=2), dtype=np.float32) # compute outdegree (= rowsum)
    D = np.nan_to_num(np.power(D,-0.5), posinf=0, neginf=0) # normalize (**-(1/2))
    D = np.asarray([np.diagflat(dd) for dd in D]) # and diagonalize
    return np.matmul(D, np.matmul(A, D))


# def make_adjacencies(particles, pt_idx=2):
#     # construct mask for real particles
#     real_p_mask = particles[:, :, pt_idx] > 0
#     adjacencies = (real_p_mask * real_p_mask.reshape(
#         real_p_mask.shape[0], real_p_mask.shape[2], real_p_mask.shape[1])).astype('float32')
#     return adjacencies

def make_adjacencies(particles):
    real_p_mask = particles[:,:,0] > 0 # construct mask for real particles
    adjacencies = (real_p_mask[:,:,np.newaxis] * real_p_mask[:,np.newaxis,:]).astype('float32')
    return adjacencies


def prepare_data_jet_based(filename, interest_feats, num_instances, start=0, end=-1):
    # set the correct background filename
    filename = filename
    data = h5py.File(filename, 'r')
    constituents = data['jetConstituentList'][start:end, ]
    # selecting only the constituents features of interest
    all_constit_features = [name.decode("utf-8")
                            for name in list(data['particleFeatureNames'])]
    interest_feats_idx = [
        all_constit_features.index(f) for f in interest_feats]
    constituents = constituents[:, :, interest_feats_idx]
    print(constituents.shape)
    jets = np.squeeze(data['fatjets'][start:end, ], axis=1)
    samples = jets_to_input_samples(constituents, jets)
    # The dataset is N_jets x N_constituents x N_features
    njet = samples.shape[0]
    if (njet > num_instances):
        samples = samples[:num_instances, :, :]
    nodes_n = samples.shape[1]
    feat_sz = samples.shape[2]
    print('Number of jets =', njet)
    print('Number of constituents (nodes) =', nodes_n)
    print('Number of features =', feat_sz)
    A = make_adjacencies(samples)
    A_tilde = normalized_adjacency(A)
    particles = normalize_features_jet_based(samples, interest_feats)
    return nodes_n, feat_sz, samples, A, A_tilde
