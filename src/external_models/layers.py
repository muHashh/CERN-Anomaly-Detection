import tensorflow as tf

''' GC layers adapted from Kipf: https://github.com/tkipf/gae/blob/0ebbe9b9a8f496eb12deb9aa6a62e7016b5a5ac3/gae/layers.py '''

class GraphConvolution(tf.keras.layers.Layer):
    
    def __init__(self, output_sz, activation=tf.keras.activations.linear, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_sz = output_sz
        self.activation = activation

    def build(self, input_shape):
        # build is invoked first time the layer is called, input_shape is based on the first argument 
        # passed to call that is stripped from args & kwargs as 'inputs': https://github.com/tensorflow/tensorflow/blob/b36436b087bd8e8701ef51718179037cccdfc26e/tensorflow/python/keras/engine/base_layer.py#L981-L982
        
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.output_sz], initializer=tf.keras.initializers.GlorotUniform())

        # TODO: add bias? (if yes, can also call base layer build directly)


    def call(self, inputs, adjacency):
        x = tf.matmul(inputs, self.kernel)
        x = tf.matmul(adjacency, x)
        return self.activation(x)

    def get_config(self):
        config = super(GraphConvolution, self).get_config()
        config.update({'output_sz': self.output_sz, 'activation': self.activation})
        return config

class GraphConvolutionRecurBias(tf.keras.layers.Layer):
    
    ''' basic graph convolution layer performing act(AXW1 + XW2 + B), nodes+neigbours and self-loop weights plus bias term '''

    def __init__(self, output_sz, activation=tf.keras.activations.linear, **kwargs):
        super(GraphConvolutionRecurBias, self).__init__(**kwargs)
        self.output_sz = output_sz
        self.activation = activation

    def build(self, input_shape):
        self.wgt1 = self.add_weight("weight_1",shape=[int(input_shape[-1]), self.output_sz], initializer=tf.keras.initializers.GlorotUniform())
        # self-loop weights
        self.wgt2 = self.add_weight("weight_2",shape=[int(input_shape[-1]), self.output_sz], initializer=tf.keras.initializers.GlorotUniform())
        self.bias = self.add_weight("bias",shape=[self.output_sz])
        

    def call(self, inputs, adjacency):
        xw1 = tf.matmul(inputs, self.wgt1)
        xw2 = tf.matmul(inputs, self.wgt2)
        axw1 = tf.matmul(adjacency, xw1)
        axw = axw1 + xw2           # add node and neighbours weighted features (self reccurency)
        layer = tf.nn.bias_add(axw, self.bias) 
        return self.activation(layer)
    

    def get_config(self):
        config = super(GraphConvolutionRecurBias, self).get_config()
        config.update({'output_sz': self.output_sz, 'activation': self.activation})
        return config


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


class InnerProductDecoder(tf.keras.layers.Layer):

    ''' inner product decoder reconstructing adjacency matrix as act(z^T z) 
        input assumed of shape [batch_sz x n_nodes x z_d]
        where 
            batch_sz can be 1 for single example feeding
            n_nodes ... number of nodes in graph
            z_d ... dimensionality of latent space
    '''

    def __init__(self, activation=tf.keras.activations.linear, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.activation = activation

    def call(self, inputs):
        perm = [0, 2, 1] if len(inputs.shape) == 3 else [1, 0]
        z_t = tf.transpose(inputs, perm=perm)
        adjacency_hat = tf.matmul(inputs, z_t)
        return self.activation(adjacency_hat)


    def get_config(self):
        config = super(InnerProductDecoder, self).get_config()
        return config
