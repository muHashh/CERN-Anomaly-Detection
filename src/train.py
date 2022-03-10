import numpy as np
import glob
import os
import h5py
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow_model_optimization as tfmot
from models import conv_ae, garnet_ae, gcn_vae, graph_ae
from external_models.graph_nn import KLWarmupCallback
from utils.preprocessing import *
import argparse

'''

Example usage: python train.py --model=graph --signals="./signals" --dataset="./dataset" --out="./output/graph" --quant_size=0 --pruning=False --latent_dim=8 --device 0

'''

model_names = {
                "cnn": conv_ae, 
                "garnet": garnet_ae, 
                "gcn": gcn_vae, 
                "graph": graph_ae,
                }


parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Model choice for training", type=str, choices=model_names.keys(), default="cnn")
parser.add_argument("--signals", help="Where the signals are loacted (HDF5 format)", type=str, default="./signals")
parser.add_argument("--dataset", help="Where the dataset is located (HDF5 format)", type=str, default="./dataset")
parser.add_argument("--out", help="Location of model output", type=str, default="./output/cnn")
parser.add_argument("--quant_size", help="Size of quantisation on model", type=int, default=0)
parser.add_argument("--pruning", help="Whether pruning is enabled or not", type=bool, default=False)
parser.add_argument("--latent_dim", help="Size of the latent space dimension", type=int, default=8)
parser.add_argument("--device", help="CUDA device for training", type=int, default=0)
args = parser.parse_args()


def train(model, signals, dataset, out, latent_dim=8, quant_size=0, pruning=False, device=0):

    ae_model = model_names[model]

    # check for output directory
    if not os.path.exists(out):
        os.makedirs(out)

    # GPU config
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    physical_devices = tf.config.list_physical_devices("GPU")
    print(physical_devices)
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # load dataset
    dataset = h5py.File(dataset+"/dataset.h5", "r")
    X_train = x_train = dataset["x_train"][()]
    X_test = x_test = dataset["x_test"][()]
    y_train = dataset["y_train"][()]
    y_test = dataset["y_test"][()]

    # define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                          patience=2, verbose=1),
    ]

    if pruning:
        callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    # get the autoencoder and prepare data
    if ae_model in {gcn_vae}:

        x_train = np.squeeze(x_train, axis=3)
        x_test = np.squeeze(x_test, axis=3)
        
        particles_bg = normalize_features(x_train)
        A_tilde_bg = normalized_adjacency(make_adjacencies(x_train))
        particles_bg_test = normalize_features(x_test)
        A_tilde_bg_test = normalized_adjacency(make_adjacencies(x_test))
        
        model = ae_model(x_train.shape[1], x_train.shape[2])

    else:
        if ae_model == garnet_ae:
            X_train = (x_train, np.ones((x_train.shape[0], 1))*x_train.shape[1])
            X_test = (x_test, np.ones((x_test.shape[0], 1))*x_test.shape[1])

        model = ae_model(quant_size=quant_size, pruning=pruning,
                         size=x_train.shape[0]+x_test.shape[0], latent_dim=latent_dim)

    # begin training
    batch_size = 2048
    n_epochs = 50

    hist = model.fit(
        x=X_train if ae_model is not gcn_vae else particles_bg,
        y=y_train if ae_model is not gcn_vae else A_tilde_bg,
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=2,
        validation_split=0.2,
        callbacks=callbacks)

    print("\nPredicting on the test dataset...")
    # Predictions
    if ae_model == gcn_vae:
        pred, _, _, _ = model((particles_bg_test, A_tilde_bg_test))
        pred = pred.numpy().astype('float32')
    else:
        pred = model.predict(X_test)

    print("\nDone!")  

    # save the model
    output = h5py.File(out + "/output.h5", 'w')
    output.create_dataset('val_loss', data=hist.history['val_loss'])
    output.create_dataset('QCD', data=x_test)

    if ae_model == gcn_vae:

        output.create_dataset('predicted_QCD', data=pred)
        output.create_dataset('loss', data=hist.history['loss'])
        
    elif ae_model != graph_ae:

        output.create_dataset('predicted_QCD', data=pred)
        output.create_dataset('loss', data=hist.history['loss'])

        model.save(out + "/model.h5")
    else:
        output.create_dataset('predicted_QCD', data=pred[0])

    print("\nPredicting on the signals...")

    for signal in glob.glob(signals+"/*"):
        signal_jets = h5py.File(signal, 'r')["jetConstituentsList"][()]

        if ae_model == garnet_ae:
            signal_jets = (signal_jets, np.ones((signal_jets.shape[0], 1))*signal_jets.shape[1])
        elif ae_model in {gcn_vae}:
            particles_signal = normalize_features(signal_jets)
            A_tilde_signal = normalized_adjacency(make_adjacencies(signal_jets))

        if ae_model != gcn_vae:
            pred_anomaly = model.predict(signal_jets)
        else:
            pred_anomaly, _, _, _, = model((particles_signal, A_tilde_signal))
            pred_anomaly = pred_anomaly.numpy().astype('float32')

        output.create_dataset('predicted_'+Path(signal).stem, data=pred_anomaly)

    output.close()

    print("\nSaved to", out)

if __name__ == "__main__":
    train(**vars(args))
