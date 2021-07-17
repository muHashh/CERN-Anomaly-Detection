import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split

def create_dataset(bg_loc, outdir):
    
    # check output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # retrieve dataset from h5 file
    h5file = h5py.File(bg_loc, "r")
    data = h5file["jetConstituentsList"][()]
    features = h5file["particleFeatureNames"][()]
    target = np.copy(data)

    h5file.close()

    # split the data and target data
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.5, shuffle=True)

    # reshpae the dataset
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], 1)
    Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], 1)
    
    h5f = h5py.File(outdir + "/dataset.h5", 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('Y_train', data=Y_train)
    h5f.create_dataset('Y_test', data=Y_test)

if __name__ == "__main__":
    create_dataset("../../data/bkg_3mln.h5", "dataset")