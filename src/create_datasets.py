import h5py
import numpy as np
import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import argparse

'''

Example usage: python create_datasets.py --qcd="../../data/bkg_3mln.h5" --signals="../../data/sig*" --qcd_out="./dataset" --signals_out="./signals" --no-scale

'''

parser = argparse.ArgumentParser()
parser.add_argument("--qcd", help="Location of background data (HDF5 format)", type=str, default="../../data/bkg_3mln.h5")
parser.add_argument("--signals", help="Location of signal data (HDF5 format)", type=str, default="../../data/sig*")
parser.add_argument("--qcd_out", help="Location of processed QCD output directory", type=str, default="./dataset/")
parser.add_argument("--signals_out", help="Location of processed signals output directory", type=str, default="./signals/")
parser.add_argument('--scale', dest='scale', action='store_true')
parser.add_argument('--no-scale', dest='scale', action='store_false')
parser.set_defaults(scale=False)
args = parser.parse_args()


def create_dataset(bg_loc, outdir, scale):
    
    # check for output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # retrieve dataset from h5 file
    h5file = h5py.File(bg_loc, 'r')
    data = h5file["jetConstituentsList"][()]
    if scale:
        data[:,:,0] = MinMaxScaler(feature_range=(0, 100)).fit_transform(data[:,:,0]) # eta
        data[:,:,1] = MinMaxScaler(feature_range=(0, 100)).fit_transform(data[:,:,1]) # phi
        # data[:,:,2] = MinMaxScaler(feature_range=(0, 100)).fit_transform(data[:,:,2]) # pT
    else:
        data[:,:,2] = data[:,:,2]/max(data[:,:,2])

    target = np.copy(data)

    h5file.close()

    # split the data and target data
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.5, shuffle=True)

    # reshpae the dataset
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)
    
    h5f = h5py.File(outdir + "/dataset.h5", 'w')
    h5f.create_dataset('x_train', data=x_train)
    h5f.create_dataset('x_test', data=x_test)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('y_test', data=y_test)

def scale_signals(signals_loc, outdir, scale):

    # check for output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for signal_loc in glob.glob(signals_loc):
        jets = h5py.File(signal_loc, 'r')["jetConstituentsList"][()]
        if scale:
            jets[:,:,0] = MinMaxScaler(feature_range=(0, 100)).fit_transform(jets[:,:,0]) # eta
            jets[:,:,1] = MinMaxScaler(feature_range=(0, 100)).fit_transform(jets[:,:,1]) # phi
            # jets[:,:,2] = MinMaxScaler(feature_range=(0, 100)).fit_transform(jets[:,:,2]) # pT
            extension = "_scaled.h5"
        else:
            jets[:,:,2] = jets[:,:,2]/max(jets[:,:,2])
            extension = ".h5"


        jets = jets.reshape(jets.shape[0], jets.shape[1], jets.shape[2], 1)

        output = h5py.File(outdir + "/" + Path(signal_loc).stem + extension, 'w')
        output.create_dataset("jetConstituentsList", data=jets)
        output.close()


if __name__ == "__main__":
    create_dataset(args.qcd, args.qcd_out, args.scale)
    scale_signals(args.signals, args.signals_out, args.scale)