import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from qkeras import QConv2D, QDense, QActivation
import tensorflow_model_optimization as tfmot
from tensorflow.keras.layers import (
    Input,
    Reshape,
    Dense,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
    Flatten,
    Activation,
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

def conv_ae(size, latent_dim=8, quant_size=0, pruning=False):
    
    int_size = QUANT_INT[quant_size]
    
    # encoder
    input_encoder = Input(shape=(16,3,1), name='encoder_input')
    x = BatchNormalization()(input_encoder)
    x = Conv2D(16, kernel_size=(3,3), use_bias=False, padding='valid')(x) if quant_size==0 \
        else QConv2D(16, kernel_size=(3,3), use_bias=False, padding='valid',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)
    
    x = AveragePooling2D(pool_size=(3, 1))(x)
    x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(32, kernel_size=(3,1), use_bias=False, padding='same',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)
    
    x = AveragePooling2D(pool_size=(3, 1))(x)
    x = Flatten()(x)

    enc = Dense(latent_dim)(x)
    encoder = Model(inputs=input_encoder, outputs=enc)
    encoder.summary()

    # decoder
    input_decoder = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(32)(input_decoder) if quant_size==0 \
        else QDense(32,
               kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)',
               bias_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(input_decoder)
    
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)
    
    x = Reshape((1,1,32))(x)
    x = UpSampling2D((4,1))(x)
    x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(32, kernel_size=(3,1), use_bias=False, padding='same',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)
    
    x = UpSampling2D((2,1))(x)
    x = Conv2D(16, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(16, kernel_size=(3,1), use_bias=False, padding='same',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)
    
    x = UpSampling2D((2,3))(x)
    dec = Conv2D(1, kernel_size=(3,3), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(1, kernel_size=(3,3), use_bias=False, padding='same',
                        kernel_quantizer='quantized_bits(16,10,0,alpha=1)')(x)
    
    decoder = Model(inputs=input_decoder, outputs=dec)
    decoder.summary() # AE
    ae_outputs = decoder(encoder(input_encoder))
    autoencoder = Model(inputs=input_encoder, outputs=ae_outputs)
    autoencoder.summary()

    if pruning:
        start_pruning = np.ceil(size*0.8/1024).astype(np.int32) * 5
        end_pruning = np.ceil(size*0.8/1024).astype(np.int32) * 15
        
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                                initial_sparsity=0.0, final_sparsity=0.5,
                                begin_step=start_pruning, end_step=end_pruning)
        encoder_pruned = tfmot.sparsity.keras.prune_low_magnitude(encoder, pruning_schedule=pruning_schedule)
        encoder = encoder_pruned
        decoder_pruned = tfmot.sparsity.keras.prune_low_magnitude(decoder, pruning_schedule=pruning_schedule)
        decoder = decoder_pruned

    # compile AE
    autoencoder.compile(optimizer=Adam(lr=3E-3, amsgrad=True), loss='mse')
    return autoencoder, encoder