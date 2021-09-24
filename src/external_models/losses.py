import tensorflow as tf
import numpy as np


### Latent Space Loss (KL-Divergence)
@tf.function
def kl_loss(z_mean, z_log_var):
    kl = 1. + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    return -0.5 * tf.reduce_mean(kl, axis=-1)

### 3D LOSS
@tf.function
def threeD_loss(inputs, outputs): #[batch_size x 100 x 3] -> [batch_size]
    expand_inputs = tf.expand_dims(inputs, 2) # add broadcasting dim [batch_size x 100 x 1 x 3]
    expand_outputs = tf.expand_dims(outputs, 1) # add broadcasting dim [batch_size x 1 x 100 x 3]
    # => broadcasting [batch_size x 100 x 100 x 3] => reduce over last dimension (eta,phi,pt) => [batch_size x 100 x 100] where 100x100 is distance matrix D[i,j] for i all inputs and j all outputs
    distances = tf.math.reduce_sum(tf.math.squared_difference(expand_inputs, expand_outputs), -1)
    # get min for inputs (min of rows -> [batch_size x 100]) and min for outputs (min of columns)
    min_dist_to_inputs = tf.math.reduce_min(distances,1)
    min_dist_to_outputs = tf.math.reduce_min(distances,2)
    return tf.math.reduce_mean(min_dist_to_inputs, 1) + tf.math.reduce_mean(min_dist_to_outputs, 1)

def threeD_loss_manual(inputs, outputs):
    distances = np.sum(np.subtract(inputs[:,:,np.newaxis,:],outputs[:,np.newaxis,:,:])**2, axis=-1)
    min_dist_to_inputs = np.min(distances,axis=1)
    min_dist_to_outputs = np.min(distances,axis=2)
    return np.sum(min_dist_to_inputs,axis=1) + np.sum(min_dist_to_outputs,axis=1)


