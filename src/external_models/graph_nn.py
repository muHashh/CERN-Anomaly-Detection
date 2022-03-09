import tensorflow as tf
from external_models.layers import *
from external_models.losses import *
import tensorflow.keras.layers as klayers


class GraphConvolutionBias(tf.keras.layers.Layer):

    ''' basic graph convolution layer performing act(AXW1 + XW2 + B), nodes+neigbours and self-loop weights plus bias term '''

    def __init__(self, output_sz, activation=tf.keras.activations.linear, **kwargs):
        super(GraphConvolutionBias, self).__init__(**kwargs)
        self.output_sz = output_sz
        self.activation = activation

    def build(self, input_shape):
        self.wgt1 = self.add_weight("weight_1",shape=[int(input_shape[-1]), self.output_sz], initializer=tf.keras.initializers.GlorotUniform())
        # self-loop weights
        self.bias = self.add_weight("bias",shape=[self.output_sz])


    def call(self, inputs, adjacency):
        xw1 = tf.matmul(inputs, self.wgt1)
        axw1 = tf.matmul(adjacency, xw1)
        layer = tf.nn.bias_add(axw1, self.bias)
        return self.activation(layer)


    def get_config(self):
        config = super(GraphConvolutionBias, self).get_config()
        config.update({'output_sz': self.output_sz, 'activation': self.activation})
        return config

class GraphAutoencoder(tf.keras.Model):

    def __init__(self, nodes_n, feat_sz, activation=tf.nn.tanh, **kwargs):
        super(GraphAutoencoder, self).__init__(**kwargs)
        self.nodes_n = nodes_n
        self.feat_sz = feat_sz
        self.input_shape_feat = [self.nodes_n, self.feat_sz]
        self.input_shape_adj = [self.nodes_n, self.nodes_n]
        self.activation = activation
        self.loss_fn = tf.nn.weighted_cross_entropy_with_logits
        self.encoder = self.build_encoder()
        self.decoder = None


    def build_encoder(self):
        ''' reduce feat_sz to 2 '''
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat
        #feat_sz-1 layers needed to reduce to R^2
        for output_sz in reversed(range(2, self.feat_sz)):
            x = GraphConvolutionBias(output_sz=output_sz, activation=self.activation)(x, inputs_adj)
        # NO activation before latent space: last graph with linear pass through activation
        x = GraphConvolutionBias(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)
        encoder = tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=x)
        encoder.summary()
        return encoder


    def call(self, inputs):
        z = self.encoder(inputs)
        adj_pred = self.decoder(z)
        return z, adj_pred

    def train_step(self, data):
        (X, adj_tilde), adj_orig = data
        # pos_weight = zero-adj / one-adj -> no-edge vs edge ratio (if more zeros than ones: > 1, if more ones than zeros < 1, e.g. for 1% of ones: 100)
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.cast(tf.math.reduce_sum(adj_orig), tf.float32)

        with tf.GradientTape() as tape:
            z, adj_pred = self((X, adj_tilde))  # Forward pass
            # Compute the loss value (binary cross entropy for a_ij in {0,1})
            loss = self.loss_fn(labels=adj_orig, logits=adj_pred, pos_weight=pos_weight) # TODO: add regularization

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        (X, adj_tilde), adj_orig = data
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.cast(tf.math.reduce_sum(adj_orig), tf.float32)

        z, adj_pred = self((X, adj_tilde), training=False)  # Forward pass
        loss = tf.math.reduce_mean(self.loss_fn(labels=adj_orig, logits=adj_pred, pos_weight=pos_weight)) # TODO: add regularization

        return {'loss' : loss}


class GCNVariationalAutoEncoder(GraphAutoencoder):

    def __init__(self, nodes_n, feat_sz, activation, latent_dim, beta_kl, kl_warmup_time, **kwargs):
        self.loss_fn_latent = kl_loss
        self.latent_dim = latent_dim
        self.kl_warmup_time = kl_warmup_time
        self.beta_kl = beta_kl
        self.beta_kl_warmup = tf.Variable(0.0, trainable=False, name='beta_kl_warmup', dtype=tf.float32)
        super(GCNVariationalAutoEncoder , self).__init__(nodes_n, feat_sz, activation, **kwargs)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat

        x = GraphConvolutionBias(output_sz=6, activation=self.activation)(x, inputs_adj)
        x = GraphConvolutionBias(output_sz=2, activation=self.activation)(x, inputs_adj)

        '''create flatten layer'''
        x = klayers.Flatten()(x) #flattened to 2 x nodes_n
        '''create dense layer #1 '''
        x = klayers.Dense(self.nodes_n, activation='relu')(x)
        print(self.nodes_n)
        ''' create dense layer #2 to make latent space params mu and sigma in last compression to feat_sz = 1 '''
        self.z_mean = klayers.Dense(self.latent_dim, activation=tf.keras.activations.linear)(x)
        self.z_log_var = klayers.Dense(self.latent_dim, activation=tf.keras.activations.linear)(x)
        batch = tf.shape(self.z_mean)[0]
        dim = tf.shape(self.z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        self.z = self.z_mean + tf.exp(0.5 * self.z_log_var) * epsilon

        encoder =  tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=[self.z, self.z_mean, self.z_log_var])
        encoder.summary()
        return encoder

    def build_decoder(self):
        inputs_feat = tf.keras.layers.Input(shape=self.latent_dim, dtype=tf.float32, name='decoder_input_latent_space')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='decoder_input_adjacency')
        out = inputs_feat

        out = klayers.Dense(self.nodes_n, activation='relu')(out)
        out = klayers.Dense(2*self.nodes_n, activation='relu')(out)
        ''' reshape to 2 x nodes_n '''
        out = tf.keras.layers.Reshape((self.nodes_n,2), input_shape=(2*self.nodes_n,))(out)
        ''' reconstruct '''
        out = GraphConvolutionBias(output_sz=6, activation=self.activation)(out, inputs_adj)
        out = GraphConvolutionBias(output_sz=self.feat_sz, activation=self.activation)(out, inputs_adj)

        decoder =  tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=out)
        decoder.summary()
        return decoder

    def call(self, inputs):
        (X, adj_orig) = inputs
        z, z_mean, z_log_var = self.encoder(inputs)
        features_out = self.decoder( (z, adj_orig) )
        return features_out, z, z_mean, z_log_var

    def train_step(self, data):
        (X, adj_orig) = data

        with tf.GradientTape() as tape:
            features_out, z, z_mean, z_log_var  = self((X, adj_orig))  # Forward pass
            # Compute the loss value ( Chamfer plus KL)
            loss_reco = tf.math.reduce_mean(threeD_loss(X,features_out))
            loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var))
            loss = loss_reco + self.beta_kl * self.beta_kl_warmup * loss_latent
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {'loss' : loss, 'loss_reco': loss_reco, 'loss_latent': loss_latent, 'beta_kl_warmup':self.beta_kl_warmup}

    def test_step(self, data):
        (X, adj_orig) = data
        features_out, z, z_mean, z_log_var = self((X, adj_orig))  # Forward pass
        loss_reco = tf.math.reduce_mean(threeD_loss(X,features_out))
        loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var))
        loss = loss_reco + self.beta_kl * self.beta_kl_warmup * loss_latent
        return {'loss' : loss, 'loss_reco': loss_reco, 'loss_latent': loss_latent}

class KLWarmupCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(KLWarmupCallback, self).__init__()
        self.beta_kl_warmup = tf.Variable(0.0, trainable=False, name='beta_kl_warmup', dtype=tf.float32)

    def on_epoch_begin(self, epoch, logs=None):
        if self.model.kl_warmup_time!=0 :
            #By design the first epoch will have a small fraction of latent loss
            kl_value = ((epoch+1)/self.model.kl_warmup_time) * (epoch < self.model.kl_warmup_time) + 1.0 * (epoch >= self.model.kl_warmup_time)
        else :
            kl_value=1
        tf.keras.backend.set_value(self.model.beta_kl_warmup, kl_value)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['beta_kl_warmup'] = tf.keras.backend.get_value(self.model.beta_kl_warmup)





