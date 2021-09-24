import tensorflow as tf
import external_models.layers as lays
import external_models.losses as losses
import tensorflow.keras.layers as klayers

@tf.function
def adjacency_loss_from_logits(adj_orig, adj_pred, pos_weight):
    # cast probability to a_ij = 1 if > 0.5 or a_ij = 0 if <= 0.5
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=adj_pred, labels=adj_orig, pos_weight=pos_weight)) 
    # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adj_pred, labels=adj_orig)) 


### Latent Space Loss (KL-Divergence)
@tf.function
def kl_loss(z_mean, z_log_var):
    kl = 1. + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    return -0.5 * tf.reduce_mean(kl, axis=-1) # multiplying mse by N -> using sum (instead of mean) in kl loss


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
        self.decoder = lays.InnerProductDecoder(activation=tf.keras.activations.linear) # if activation sigmoid -> return probabilities from logits


    def build_encoder(self):
        ''' reduce feat_sz to 2 '''
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat
        #feat_sz-1 layers needed to reduce to R^2 
        for output_sz in reversed(range(2, self.feat_sz)):
            x = lays.GraphConvolutionBias(output_sz=output_sz, activation=self.activation)(x, inputs_adj)
        # NO activation before latent space: last graph with linear pass through activation
        x = lays.GraphConvolutionBias(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)
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




class GraphVariationalAutoencoder(GraphAutoencoder):
    
    def __init__(self, nodes_n, feat_sz, activation, beta=1.0, **kwargs):
        super(GraphVariationalAutoencoder, self).__init__(nodes_n, feat_sz, activation, **kwargs)
        self.loss_fn_latent = kl_loss
        self.beta = beta

    def build_encoder(self):

        ''' reduce feat_sz to 2 '''
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat

        for output_sz in reversed(range(2, self.feat_sz)):
            x = lays.GraphConvolutionBias(output_sz=output_sz, activation=self.activation)(x, inputs_adj)

        ''' make latent space params mu and sigma in last compression to feat_sz = 1 '''
        self.z_mean = lays.GraphConvolutionBias(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)
        self.z_log_var = lays.GraphConvolutionBias(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)

        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(self.z_mean)[0], self.nodes_n, 1))
        self.z = self.z_mean +  epsilon * tf.exp(0.5 * self.z_log_var)

        return tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=[self.z, self.z_mean, self.z_log_var])
    
    
    def call(self, inputs):
        # import ipdb; ipdb.set_trace()
        z, z_mean, z_log_var = self.encoder(inputs)
        adj_pred = self.decoder(z)
        return z, z_mean, z_log_var, adj_pred
    
    def train_step(self, data):
        (X, adj_tilde), adj_orig = data
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.cast(tf.math.reduce_sum(adj_orig), tf.float32)


        with tf.GradientTape() as tape:
            z, z_mean, z_log_var, adj_pred = self((X, adj_tilde))  # Forward pass
            # Compute the loss value (binary cross entropy for a_ij in {0,1})
            loss_reco = tf.math.reduce_mean(self.loss_fn(labels=adj_orig, logits=adj_pred, pos_weight=pos_weight), axis=(1,2)) # TODO: add regularization
            loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var), axis=1)
            loss = loss_reco + loss_latent

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {'loss' : loss_reco+loss_latent, 'loss_reco': loss_reco, 'loss_latent': loss_latent}


    def test_step(self, data):
        (X, adj_tilde), adj_orig = data
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.cast(tf.math.reduce_sum(adj_orig), tf.float32)

        z, z_mean, z_log_var, adj_pred = self((X, adj_tilde))  # Forward pass
        # Compute the loss value (binary cross entropy for a_ij in {0,1})
        loss_reco =  tf.math.reduce_mean(self.loss_fn(labels=adj_orig, logits=adj_pred, pos_weight=pos_weight)) # TODO: add regularization
        loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var))
        return {'loss' : loss_reco+loss_latent, 'loss_reco': loss_reco, 'loss_latent': loss_latent}


class GCNVariationalAutoEncoder(GraphAutoencoder):
    
    def __init__(self, nodes_n, feat_sz, activation, latent_dim, beta_kl, kl_warmup_time, **kwargs):
        self.loss_fn_latent = losses.kl_loss
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

     #   x = lays.GraphConvolutionBias(output_sz=5, activation=self.activation)(x, inputs_adj)
        x = lays.GraphConvolutionBias(output_sz=6, activation=self.activation)(x, inputs_adj)
        x = lays.GraphConvolutionBias(output_sz=2, activation=self.activation)(x, inputs_adj)
     #   for output_sz in reversed(range(2, self.feat_sz)):
     #       x = lays.GraphConvolutionBias(output_sz=output_sz, activation=self.activation)(x, inputs_adj) #right now size is 2 x nodes_n

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
     #   for output_sz in range(2+1, self.feat_sz+1): #TO DO: none of this should be hardcoded , to be fixed
     #       out = lays.GraphConvolutionBias(output_sz=output_sz, activation=self.activation)(out, inputs_adj)
        out = lays.GraphConvolutionBias(output_sz=6, activation=self.activation)(out, inputs_adj)
     #   out = lays.GraphConvolutionBias(output_sz=5, activation=self.activation)(out, inputs_adj)
        out = lays.GraphConvolutionBias(output_sz=self.feat_sz, activation=self.activation)(out, inputs_adj)

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
            loss_reco = tf.math.reduce_mean(losses.threeD_loss(X,features_out))
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
        loss_reco = tf.math.reduce_mean(losses.threeD_loss(X,features_out))
        loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var))
        loss = loss_reco + self.beta_kl * self.beta_kl_warmup * loss_latent
        return {'loss' : loss, 'loss_reco': loss_reco, 'loss_latent': loss_latent}   


class KLWarmupCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(KLWarmupCallback, self).__init__()
        self.beta_kl_warmup = tf.Variable(0.0, trainable=False, name='beta_kl_warmup', dtype=tf.float32)

    def on_epoch_begin(self, epoch, logs=None):
        kl_value = (epoch/self.model.kl_warmup_time) * (epoch <= self.model.kl_warmup_time) + 1.0 * (epoch > self.model.kl_warmup_time)
        tf.keras.backend.set_value(self.model.beta_kl_warmup, kl_value)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['beta_kl_warmup'] = tf.keras.backend.get_value(self.model.beta_kl_warmup)





