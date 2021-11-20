import numpy as np
import glob
import os
import h5py
import tensorflow as tf
from pathlib import Path
import argparse
from tensorflow.keras.models import load_model
from external_models.garnet import GarNet

'''

This should only be used to predict on different datasets with already trained models, as the predictions are already created when the training occurs

'''

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Location of the model.h5 file created during training", type=str)
parser.add_argument("--signals", help="Where the signals are loacted (HDF5 format)", type=str, default="./signals")
parser.add_argument("--dataset", help="Where the dataset is located (HDF5 format)", type=str, default="./dataset")
parser.add_argument("--out", help="Name and location of the output H5 file", type=str)
args = parser.parse_args()


def predict(model, signals, dataset, out):
    
    # check for output directory
    if not os.path.exists(out):
        os.makedirs(out)
    
    # GPU config
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = load_model(model, custom_objects={"GarNet": GarNet})

    with h5py.File(out+"/predictions.h5", "w") as output:

        dataset = h5py.File(dataset+"/dataset.h5", "r")
        x_test =  dataset["y_test"][()] + dataset["x_test"][()]
        X_test = (x_test, np.ones((x_test.shape[0], 1))*x_test.shape[1])

        prediction = model.predict(X_test)
        output.create_dataset('predicted_QCD', data=prediction)
        output.create_dataset('QCD', data=x_test)

        for signal in glob.glob(signals+"/*"):
            signal_jets = h5py.File(signal, 'r')["jetConstituentsList"][()]

            signal_jets = (signal_jets, np.ones((signal_jets.shape[0], 1))*signal_jets.shape[1])

            pred_anomaly = model.predict(signal_jets)
            output.create_dataset('predicted_'+Path(signal).stem, data=pred_anomaly)

if __name__ == "__main__":
    predict(**vars(args))
