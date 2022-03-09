import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from qkeras import QConv2D, QDense, QActivation
import tensorflow_model_optimization as tfmot
from external_models.garnet_old import GarNet
import external_models.graph_nn as graph
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Reshape,
    Dense,
    Conv2D,
    AveragePooling2D,
    UpSampling2D,
    BatchNormalization,
    Flatten,
    Activation,
    LeakyReLU
)

# number of integer bits for each bit width
QUANT_INT = {
    0: 0,
    2: 1,
    4: 2,
    6: 2,
    8: 3,
    10: 3,
    12: 4,
    14: 4,
    16: 6
}


"""

Custom loss functions

"""

def mse_loss(inputs, outputs):
    return tf.math.reduce_mean(tf.math.square(outputs-inputs), axis=-1)

def make_mse(inputs, outputs):

    inputs = tf.cast(inputs, dtype=tf.float32)

    # trick with phi
    outputs[:,:,1].value = np.pi*tf.math.tanh(outputs[:,:,1])
    # trick with phi
    outputs[:,:,0].value = 4.0*tf.math.tanh(outputs[:,:,0])

    # use both tricks
    # outputs = tf.concat([outputs_eta[:,:,0], outputs_phi[:,:,1], outputs[:,:,2]], axis=1)
    # outputs = tf.reshape(outputs, (tf.shape(outputs)[0],) + (16, 3))
    
    # mask zero features
    mask = tf.math.not_equal(inputs,0)
    mask = tf.cast(mask, tf.float32)
    outputs = mask * outputs

    loss = mse_loss(inputs, outputs)
    loss = tf.math.reduce_mean(loss, axis=0) # average over batch
    return loss


def chamfer_loss(inputs, outputs):  # [batch_size x 100 x 3] -> [batch_size]
    # add broadcasting dim [batch_size x 100 x 1 x 3]
    expand_inputs = tf.expand_dims(inputs, 2)
    # add broadcasting dim [batch_size x 1 x 100 x 3]
    expand_outputs = tf.expand_dims(outputs, 1)
    # => broadcasting [batch_size x 100 x 100 x 3] => reduce over last dimension (eta,phi,pt) => [batch_size x 100 x 100] where 100x100 is distance matrix D[i,j] for i all inputs and j all outputs
    distances = tf.math.reduce_sum(
        tf.math.squared_difference(expand_inputs, expand_outputs), -1)
    # get min for inputs (min of rows -> [batch_size x 100]) and min for outputs (min of columns)
    min_dist_to_inputs = tf.math.reduce_min(distances, 1)
    min_dist_to_outputs = tf.math.reduce_min(distances, 2)
    return tf.math.reduce_mean(min_dist_to_inputs, 1) + tf.math.reduce_mean(min_dist_to_outputs, 1)



"""

Models 

"""

def conv_ae(size=0, latent_dim=8, quant_size=0, pruning=False):

    int_size = QUANT_INT[quant_size]

    # encoder
    input_encoder = Input(shape=(16, 3, 1), name='encoder_input')
    x = BatchNormalization()(input_encoder)
    x = Conv2D(16, kernel_size=(3, 3), use_bias=False, padding='valid')(x) if quant_size == 0 \
        else QConv2D(16, kernel_size=(3, 3), use_bias=False, padding='valid',
                     kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)

    x = Activation('relu')(x) if quant_size == 0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = AveragePooling2D(pool_size=(3, 1))(x)
    x = Conv2D(32, kernel_size=(3, 1), use_bias=False, padding='same')(x) if quant_size == 0 \
        else QConv2D(32, kernel_size=(3, 1), use_bias=False, padding='same',
                     kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)

    x = Activation('relu')(x) if quant_size == 0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = AveragePooling2D(pool_size=(3, 1))(x)
    x = Flatten()(x)

    enc = Dense(latent_dim)(x)
    encoder = Model(inputs=input_encoder, outputs=enc)
    encoder.summary()

    # decoder
    input_decoder = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(32)(input_decoder) if quant_size == 0 \
        else QDense(32,
                    kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)',
                    bias_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(input_decoder)

    x = Activation('relu')(x) if quant_size == 0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = Reshape((1, 1, 32))(x)
    x = UpSampling2D((4, 1))(x)
    x = Conv2D(32, kernel_size=(3, 1), use_bias=False, padding='same')(x) if quant_size == 0 \
        else QConv2D(32, kernel_size=(3, 1), use_bias=False, padding='same',
                     kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)

    x = Activation('relu')(x) if quant_size == 0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = UpSampling2D((2, 1))(x)
    x = Conv2D(16, kernel_size=(3, 1), use_bias=False, padding='same')(x) if quant_size == 0 \
        else QConv2D(16, kernel_size=(3, 1), use_bias=False, padding='same',
                     kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)

    x = Activation('relu')(x) if quant_size == 0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = UpSampling2D((2, 3))(x)
    dec = Conv2D(1, kernel_size=(3, 3), use_bias=False, padding='same')(x) if quant_size == 0 \
        else QConv2D(1, kernel_size=(3, 3), use_bias=False, padding='same',
                     kernel_quantizer='quantized_bits(16,10,0,alpha=1)')(x)

    decoder = Model(inputs=input_decoder, outputs=dec)
    decoder.summary()
    
    ae_outputs = decoder(encoder(input_encoder))
    autoencoder = Model(inputs=input_encoder, outputs=ae_outputs)
    autoencoder.summary()

    if pruning:
        start_pruning = np.ceil(size*0.8/1024).astype(np.int32) * 5
        end_pruning = np.ceil(size*0.8/1024).astype(np.int32) * 15

        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=0.5,
            begin_step=start_pruning, end_step=end_pruning)
        encoder = tfmot.sparsity.keras.prune_low_magnitude(
            encoder, pruning_schedule=pruning_schedule)
        decoder = tfmot.sparsity.keras.prune_low_magnitude(
            decoder, pruning_schedule=pruning_schedule)

    # compile AE
    autoencoder.compile(optimizer=Adam(lr=1e-3, amsgrad=True), loss='mse')
    return autoencoder


def garnet_ae(size=0, latent_dim=8, quant_size=0, pruning=False):

    int_size = QUANT_INT[quant_size]

    # model inputs
    x = Input(shape=(16, 3))
    n = Input(shape=(1), dtype='uint16')
    inputs = [x, n]

    # model definition
    encoder = GarNet(16, 16*2, 2, simplified=True, collapse='mean', input_format='xn',
               output_activation='relu', name='garnet_encoder1', quantize_transforms=False)(inputs) if quant_size <= 0 \
                   else GarNet(16, 16*2, 2, simplified=True, collapse='mean', input_format='xn',
                            output_activation='relu', name='garnet_encoder1', quantize_transforms=True, total_bits=quant_size, int_bits=int_size)(inputs)
    encoder = Reshape((16,2))(encoder)
    # encoder = tf.reshape(encoder, (tf.shape(encoder)[0],) + (16, 2))
    encoder = GarNet(16, 16, 1, simplified=True, collapse='mean', input_format='xn',
               output_activation='relu', name='garnet_encoder2', quantize_transforms=False)([encoder, n]) if quant_size <= 0 \
                   else GarNet(16, 16, 1, simplified=True, collapse='mean', input_format='xn',
                            output_activation='relu', name='garnet_encoder2', quantize_transforms=True, total_bits=quant_size, int_bits=int_size)([encoder, n])
    encoder = Reshape((16,1))(encoder)
    # encoder = tf.reshape(encoder, (tf.shape(encoder)[0],) + (16, 1))

    decoder = GarNet(16, 16*2, 1, simplified=True, collapse='mean', input_format='xn',
               output_activation='relu', name='garnet_decoder1', quantize_transforms=False)([encoder, n]) if quant_size <= 0 \
                   else GarNet(16, 16*2, 1, simplified=True, collapse='mean', input_format='xn',
                            output_activation='relu', name='garnet_decoder1', quantize_transforms=True, total_bits=quant_size, int_bits=int_size)([encoder, n])
    decoder = Reshape((16,2))(decoder)
    # decoder = tf.reshape(decoder, (tf.shape(decoder)[0],) + (16, 2))
    decoder = GarNet(16, 16*3, 2, simplified=True, collapse='mean', input_format='xn',
               output_activation='linear', name='garnet_decoder2', quantize_transforms=False)([decoder, n]) if quant_size <= 0 \
                   else GarNet(16, 16*3, 2, simplified=True, collapse='mean', input_format='xn',
                            output_activation='linear', name='garnet_decoder2', quantize_transforms=True, total_bits=quant_size, int_bits=int_size)([decoder, n])
    decoder = Reshape((16,3))(decoder)
    # decoder = tf.reshape(decoder, (tf.shape(decoder)[0],) + (16, 3))

    # build model
    model = Model(inputs=[x, n], outputs=decoder)

    # compile model with adam and mean square error
    model.compile(optimizer=Adam(lr=1e-4, amsgrad=True), loss="mse")
    model.summary()

    return model


def graph_ae(nodes_n, feat_sz):

    model = graph.GraphAutoencoder(nodes_n=nodes_n, feat_sz=feat_sz)
    model.compile(optimizer=Adam(lr=3e-3, amsgrad=True), loss="mse")

    return model


def gcn_vae(nodes_n, feat_sz):

    model = graph.GCNVariationalAutoEncoder(nodes_n=nodes_n, feat_sz=feat_sz,
                                            activation=tf.nn.tanh, latent_dim=8, beta_kl=0.5, kl_warmup_time=1)
    model.compile(optimizer=Adam(lr=1e-2), loss="mse")

    return model


if __name__ == "__main__":
    garnet_ae()
