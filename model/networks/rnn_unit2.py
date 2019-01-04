from model.networks.base_network import *
from util.sparse_util import *

class RNNModel(BaseModel):
    def __init__(self, params, input_size, num_classes, seed, init):
        super(RNNModel, self).__init__(params)

        self.Network['Dummy'] = []
        self._input_size = input_size

        for ii in range(len(self.params.rnn_r_hidden_seq)):
            init_data = \
                {'w_init_method': 'normc', 'w_init_para': {'stddev':1.0}}

            if self.params.rnn_bidirectional:
                pass
            elif self.params.rnn_dilated:
                pass
            else:
                self.Network['Dummy'].append(SparseDummyRecurrentNetwork(
                    scope='rnn'+str(ii), activation_type=self.params.rnn_r_act_seq[ii],
                    normalizer_type=self.params.rnn_r_norm_seq[ii],
                    recurrent_cell_type=self.params.rnn_cell_type,
                    train=True, hidden_size=self.params.rnn_r_hidden_seq[ii],
                    input_depth=input_size, num_unitwise=self.params.num_unitwise
                ))
                input_size = self.params.rnn_r_hidden_seq[ii]

        network_shape = [self.params.rnn_r_hidden_seq[-1]] + \
            self.params.rnn_l_hidden_seq + [num_classes]

        num_layer = len(network_shape) - 1
        act_type = \
            self.params.rnn_l_act_seq + [None]
        norm_type = \
            self.params.rnn_l_norm_seq + [None]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'xavier', 'w_init_para': {'uniform': False},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )
        self.Network['Linear'] = SparseMLP(
            dims=network_shape, scope='mlp',
            sparsity=self.params.mlp_sparsity,
            activation_type=act_type, normalizer_type=norm_type,
            train=True, init_data=init_data, seed=seed
        )

    def build_sparse(self, sparse_var):
        self.Network['Recurrent'] = []
        input_size = self._input_size
        self.initialize_op = []

        for ii in range(len(self.params.rnn_r_hidden_seq)):
            sparse = sparse_var[ii]

            if self.params.rnn_bidirectional:
                pass
            elif self.params.rnn_dilated:
                pass
            else:
                self.Network['Recurrent'].append(SparseRecurrentNetwork(
                    scope='rnn' + str(ii), activation_type=self.params.rnn_r_act_seq[ii],
                    normalizer_type=self.params.rnn_r_norm_seq[ii],
                    recurrent_cell_type=self.params.rnn_cell_type,
                    train=True, hidden_size=self.params.rnn_r_hidden_seq[ii],
                    input_depth=input_size, sparse_list=sparse
                ))

                input_size = self.params.rnn_r_hidden_seq[ii]

                self.initialize_op.append(self.Network['Recurrent'][ii].initialize_op)

    def __call__(self, input):
        self.Tensor['Intermediate'] = [None for _ in self.Network['Recurrent']]
        for i, network in enumerate(self.Network['Recurrent']):
            self.Tensor['Intermediate'][i] = network(input)[0]
            input = self.Tensor['Intermediate'][i]
        self.Tensor['Predictions'] = self.Network['Linear'](self.Tensor['Intermediate'][-1])
        return self.Tensor['Predictions']

    def unit(self, input):
        # gotta fix this for multilayer
        self.Tensor['Temp'] = [None for _ in self.Network['Dummy']]
        for i, network in enumerate(self.Network['Dummy']):
            self.Tensor['Temp'][i] = network(input)[0]
            input = self.Tensor['Temp'][i]
        self.Tensor['Unit_Pred'] = self.Network['Linear'](input)
        return self.Tensor['Unit_Pred']

    def get_dummy_variables(self):
        dummy_weights = []
        dummy_biases = []
        for net in self.Network['Dummy']:
            dummy_biases.append(net.bias)
            dummy_weights.append(net.weight)
        return dummy_weights, dummy_biases
