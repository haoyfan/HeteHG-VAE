from layers import *
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


class Model(object):

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

class HeteHG_VAE(Model):

    def __init__(self, placeholders, node_types, main_type,
                 num_nodes, num_event,
                 **kwargs):

        super(HeteHG_VAE, self).__init__(**kwargs)

        self.inputs_node = {}
        self.inc_node_norm = {}
        self.inc_orig = {}
        self.num_nodes = num_nodes
        self.num_event = num_event
        self.placeholders = placeholders
        self.dropout = placeholders['dropout']
        self.node_types = node_types
        self.main_type = main_type
        self.node_embed01 = {}
        self.edge_embed01 = {}
        self.z_node_mean = {}
        self.z_node_log_std = {}
        self.z_edge_mean = {}
        self.z_edge_log_std = {}

        self.z_node = {}
        self.z_edge = {}


        self.init_var()
        self.build()

    def init_var(self):
        for node_type in self.node_types:
            if node_type==self.main_type:
                continue
            self.inc_orig[node_type] = self.placeholders['inc_orig_'+node_type]

    def _build(self):

        for node_type in self.node_types:
            if node_type==self.main_type:
                continue

            self.node_embed01[node_type] = Dense(input_dim=self.num_event,
                                              output_dim=FLAGS.hidden1,
                                              act=tf.nn.tanh,
                                              sparse_inputs=True,
                                              dropout=self.dropout)(self.inc_orig[node_type])


            self.z_node_mean[node_type] = Dense(input_dim=FLAGS.hidden1,
                                     output_dim=FLAGS.hidden2,
                                     act=lambda x: x,
                                     dropout=self.dropout)(self.node_embed01[node_type])

            self.z_node_log_std[node_type] = Dense(input_dim=FLAGS.hidden1,
                                     output_dim=FLAGS.hidden2,
                                     act=lambda x: x,
                                     dropout=self.dropout)(self.node_embed01[node_type])

            self.z_node[node_type] = self.z_node_mean[node_type] + \
                                     tf.random_normal([self.num_nodes[node_type], FLAGS.hidden2]) * tf.exp(self.z_node_log_std[node_type])

        for node_type in self.node_types:
            if node_type==self.main_type:
                continue
            self.edge_embed01[node_type] = Dense(input_dim=self.num_nodes[node_type],
                                 output_dim=FLAGS.hidden1,
                                 act=tf.nn.tanh,
                                 sparse_inputs=True,
                                 dropout=self.dropout)(tf.sparse_transpose(self.inc_orig[node_type]))


        # Hyperedge attention
        edge_embed_list = []
        for v in self.edge_embed01.values():
            edge_embed_list.append(tf.expand_dims(v, axis=1))

        edge_embeds = tf.concat(edge_embed_list, axis=1)
        self.edge_embed, self.alphas = HyperedgeAttention(input_dim=FLAGS.hidden1, output_dim=FLAGS.hidden1//2)(edge_embeds)

        self.z_edge_mean = Dense(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            act=lambda x: x,
                                            dropout=self.dropout)(self.edge_embed)

        self.z_edge_log_std = Dense(input_dim=FLAGS.hidden1,
                                               output_dim=FLAGS.hidden2,
                                               act=lambda x: x,
                                               dropout=self.dropout)(self.edge_embed)

        self.z_edge = self.z_edge_mean + tf.random_normal(
            [self.num_event, FLAGS.hidden2]) * tf.exp(
            self.z_edge_log_std)

        self.reconstructions = InnerDecoder(input_dim=FLAGS.hidden2,
                                            act=lambda x: x,
                                            # act=tf.nn.sigmoid,
                                            logging=self.logging)((self.z_node, self.z_edge))
        self.reconstructions_mean = InnerDecoder(input_dim=FLAGS.hidden2,
                                            act=lambda x: x,
                                                 logging=self.logging)((self.z_node_mean, self.z_edge_mean))

