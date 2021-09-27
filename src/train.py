import numpy as np
import glob
import os
import h5py
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import MinMaxScaler
from models import conv_ae, garnet_ae, graph_ae
import argparse

'''

Example usage: python train.py --model=graph --signal=./signal/* --dataset=./datasetdatset.h5 --outdir=./output/graph --quant_size=0 --pruning=False --latent_dim=8

'''

model_names = {"cnn": conv_ae, "garnet": garnet_ae, "graph": graph_ae}

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Model choice for training", type=str, choices=model_names.keys(), default="cnn")
parser.add_argument("--signals", help="Where the signals are loacted (HDF5 format)", type=str, default="./signals/*")
parser.add_argument("--dataset", help="Where the dataset is located (HDF5 format)", type=str, default="./dataset/dataset.h5")
parser.add_argument("--outdir", help="Location of model output", type=str, default="./output/cnn")
parser.add_argument("--quant_size", help="Size of quantisation on model", type=int, default=0)
parser.add_argument("--pruning", help="Whether pruning is enabled or not", type=bool, default=False)
parser.add_argument("--latent_dim", help="Whether pruning is enabled or not", type=int, default=8)
args = parser.parse_args()


def make_adjacencies(particles, pt_idx=2):
    real_p_mask = particles[:,:, pt_idx] > 0 # construct mask for real particles
    adjacencies = (real_p_mask * \
                    real_p_mask.reshape(real_p_mask.shape[0],real_p_mask.shape[2],real_p_mask.shape[1])).astype('float32')
    return adjacencies

def normalized_adjacency(A):
    D = np.array(np.sum(A, axis=2), dtype=np.float32) # compute outdegree (= rowsum)
    D = np.nan_to_num(np.power(D,-0.5), posinf=0, neginf=0) # normalize (**-(1/2))
    D = np.asarray([np.diagflat(dd) for dd in D]) # and diagonalize
    return np.matmul(D, np.matmul(A, D))

def train(model, signals, dataset, outdir, latent_dim=8, quant_size=0, pruning=False):

    ae_model = model_names[model]

    # check for output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # GPU config
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # load dataset
    dataset = h5py.File(dataset, "r")
    X_train = x_train = dataset["x_train"][()]
    X_test = x_test = dataset["x_test"][()]
    y_train = dataset["y_train"][()]
    y_test = dataset["y_test"][()]

    # define callbacks
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),
    ]
    
    if pruning:
        callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    # get the autoencoder
    if ae_model == graph_ae:
        model = ae_model(x_test.shape[1], x_test.shape[2])
        A_train = y_train = make_adjacencies(x_train)
        A_test = y_test = make_adjacencies(x_test)
        X_train = (x_train, normalized_adjacency(A_train))
        X_test = (x_test, normalized_adjacency(A_test))
    else:
        if ae_model == garnet_ae:
            X_train = (x_train, np.ones((x_train.shape[0],1))*latent_dim)
            X_test = (x_test, np.ones((x_test.shape[0],1))*latent_dim)

        model = ae_model(quant_size=quant_size, pruning=pruning, 
                                        size=2*x_train[0].shape[0], latent_dim=latent_dim)

    # begin training
    batch_size = 128
    n_epochs = 50

    hist = model.fit(
            x=X_train,
            y=y_train,
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=2,
            validation_split=0.2,
            callbacks=callbacks)
    
    # Predictions
    pred = model.predict(X_test)

    # save the model
    output = h5py.File(outdir + "/output.h5", 'w')
    output.create_dataset('val_loss', data=hist.history['val_loss'])
    output.create_dataset('QCD', data=x_test)
    output.create_dataset('predicted_QCD', data=pred)

    if ae_model != graph_ae:
        output.create_dataset('loss', data=hist.history['loss'])

        modelJson = model.to_json()
        with open (outdir + "/savedModel.json", "w") as f:
            f.write(modelJson)
        model.save_weights(outdir + "/savedWeights.h5")
    
    for signal_loc in glob.glob(signals):
        signal_jets = h5py.File(signal_loc, 'r')["jetConstituentsList"][()]

        if ae_model == garnet_ae:
            signal_jets = (signal_jets, np.ones((signal_jets.shape[0],1))*latent_dim)
        elif ae_model == graph_ae:
            signal_jets = (signal_jets, normalized_adjacency(make_adjacencies(signal_jets)))

        pred_anomaly = model.predict(signal_jets)
        output.create_dataset('predicted_'+Path(signal_loc).stem, data=pred_anomaly)


if __name__ == "__main__":
    train(**vars(args))
    
