import numpy as np
import pandas as pd
import glob
import h5py
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import (
    Lambda,
    Input,
    Dense,
    LeakyReLU,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Reshape,
    Activation,
    Concatenate,
    Cropping1D
)

def create_dataset(location):
    
    # retrieve dataset from h5 file
    h5file = h5py.File(location, "r")
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

    return X_train, X_test, Y_train, Y_test

def create_conv_ae(latent_dim=8):

    # encoder
    input_encoder = Input(shape=(16,3,1), name='encoder_input')
    x = BatchNormalization()(input_encoder)
    x = Conv2D(16, kernel_size=(3,3), use_bias=False, padding='valid')(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(3, 1))(x)
    x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(3, 1))(x)
    x = Flatten()(x)

    enc = Dense(latent_dim)(x)
    encoder = Model(inputs=input_encoder, outputs=enc)
    encoder.summary()

    # decoder
    input_decoder = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(32)(input_decoder) 
    x = Activation('relu')(x) 
    x = Reshape((1,1,32))(x)
    x = UpSampling2D((4,1))(x)
    x = Conv2D(16, kernel_size=(3,1), use_bias=False, padding='same')(x) 
    x = Activation('relu')(x)
    x = UpSampling2D((4,3))(x)

    dec = Conv2D(1, kernel_size=(3,3), use_bias=False, padding='same')(x)
    decoder = Model(inputs=input_decoder, outputs=dec)
    decoder.summary() # AE
    ae_outputs = decoder(encoder(input_encoder))
    autoencoder = Model(inputs=input_encoder, outputs=ae_outputs)
    autoencoder.summary()

    # compile AE
    autoencoder.compile(optimizer=Adam(lr=3E-3, amsgrad=True), loss='mse')
    return autoencoder, encoder

def train():

    # GPU config
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # load dataset and split into test and train files
    X_train, X_test, Y_train, Y_test = create_dataset("../../data/bkg_3mln.h5")

    # define callbacks
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
    ]

    # get the CNN autoencoder
    model, encoder = create_conv_ae()

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

    pred = model.predict(X_test)
    encoded = encoder.predict(X_test)

    # save the model
    modelJson = model.to_json()

    with open ("savedModel.json", "w") as f:
        f.write(modelJson)
    model.save_weights("savedWeights.h5")

    h5f = h5py.File("output.h5", 'w')
    h5f.create_dataset('loss', data=hist.history['loss'])
    h5f.create_dataset('val_loss', data=hist.history['val_loss'])
    h5f.create_dataset('QCD', data=Y_test)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('predicted_QCD', data=pred)
    h5f.create_dataset('encoded_mean_QCD', data=encoded[0])
    h5f.create_dataset('encoded_logvar_QCD', data=encoded[1])
    h5f.create_dataset('encoded_z_QCD', data=encoded[2])




if __name__ == "__main__":
    train()
