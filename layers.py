import tensorflow as tf
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS
latent_dim = 128
hidden_decoder_dim = 512

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


class Layer(object):

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    def __init__(self, input_dim, output_dim, adj, dropout = 0., act = tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name = "weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout = 0., act = tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name = "weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs




class Dense(Layer):

    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=tf.nn.relu, placeholders=None, bias=True,
                 sparse_inputs=False, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable('weights', shape=(input_dim, output_dim),
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
            if self.bias:
                self.vars['bias'] = tf.Variable(tf.zeros([output_dim], dtype=tf.float32), name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        if self.sparse_inputs:
            output = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        else:
            output = tf.matmul(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)



class HyperedgeAttention(Layer):
    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=tf.nn.tanh, **kwargs):
        super(HyperedgeAttention, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable('weights', shape=(input_dim, output_dim),
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))

            self.vars['bias'] = tf.Variable(tf.zeros([output_dim], dtype=tf.float32), name='bias')
            self.vars['attn_weights'] = tf.Variable(tf.zeros([output_dim], dtype=tf.float32), name='attn_weights')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = self.act(tf.tensordot(x, self.vars['weights'], axes=1) + self.vars['bias'])
        x = tf.tensordot(x, self.vars['attn_weights'], axes=1, name='x_attn')
        alphas = tf.nn.softmax(x, name='alphas')
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        return output, alphas



class InnerProductDecoder(Layer):
    def __init__(self, dropout = 0., act = tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def bais_variable_glorot(output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (output_dim))
    initial = tf.random_uniform([output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)



class InnerDecoder(Layer):
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.input_dim = input_dim
        self.act = act

    def _call(self, inputs):
        z_node, z_edge = inputs
        outputs = {}

        for node_type in z_node.keys():
            z_node[node_type] = tf.nn.dropout(z_node[node_type], 1 - self.dropout)
            z_a_t = tf.transpose(tf.nn.dropout(z_edge, 1 - self.dropout))
            y = tf.matmul(z_node[node_type], z_a_t)
            outputs[node_type] = self.act(y)

        return outputs

