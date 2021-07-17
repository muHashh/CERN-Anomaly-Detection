import numpy as np
import pandas as pd
import glob
import os
import h5py
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import StandardScaler
from model import conv_ae

def train(anomalies_loc, dataset_loc, outdir, quant_size=0, pruning=False):
    
    # check output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # GPU config
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # load dataset
    h5f = h5py.File(dataset_loc, "r")
    X_train = h5f["X_train"][()]
    X_test = h5f["X_test"][()]
    Y_train = h5f["Y_train"][()]
    Y_test = h5f["Y_test"][()]

    # define callbacks
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),
    ]
    
    if pruning:
        callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    # get the CNN autoencoder
    model, encoder = conv_ae(quant_size=quant_size, pruning=pruning, size=X_train.shape[0]+X_test.shape[0])

    # begin training
    batch_size = 1024
    n_epochs = 20

    hist = model.fit(
            x=X_train,
            y=Y_train,
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=2,
            validation_split=0.2,
            callbacks=callbacks)
    
    # Predictions
    pred = model.predict(X_test)
    encoded = encoder.predict(X_test)

    # save the model
    modelJson = model.to_json()

    with open (outdir + "/savedModel.json", "w") as f:
        f.write(modelJson)
    model.save_weights(outdir + "/savedWeights.h5")

    h5f = h5py.File(outdir + "/output.h5", 'w')
    h5f.create_dataset('loss', data=hist.history['loss'])
    h5f.create_dataset('val_loss', data=hist.history['val_loss'])
    h5f.create_dataset('QCD', data=Y_test)
    h5f.create_dataset('predicted_QCD', data=pred)
    h5f.create_dataset('encoded_mean_QCD', data=encoded[0])
    h5f.create_dataset('encoded_logvar_QCD', data=encoded[1])
    h5f.create_dataset('encoded_z_QCD', data=encoded[2])
    
    for f in glob.glob(anomalies_loc):
        pred_anomaly = model.predict(h5py.File(f, 'r')["jetConstituentsList"][()])
        h5f.create_dataset('predicted_'+os.path.basename(f)[:-3], data=pred_anomaly)


if __name__ == "__main__":
#     for bw in range(2,17,2):
#         train(anomalies_loc="../../data/sig*", dataset_loc="dataset/dataset.h5", outdir="output_qp"+str(bw), quant_size=bw, pruning=True)
    
    train(anomalies_loc="../../data/sig*", dataset_loc="dataset/dataset.h5", outdir="output", quant_size=0, pruning=False)

    
