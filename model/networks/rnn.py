from util.network_util import *
import tensorflow as tf
from model.networks.base_network import *

class RNNModel(BaseModel):
    def __init__(self, params, input_size, num_classes, seed, init):
        super(RNNModel, self).__init__(params)

        init_data = \
            {'w_init_method': 'normal', 'w_init_para': {'mean': 0,
                'stddev':1/self.params.rnn_r_hidden}}

        if self.params.rnn_bidirectional:
            pass
        elif self.params.rnn_dilated:
            pass
        else:
            self.Network['Recurrent'] = Recurrent_Network_with_mask(
                scope='rnn', activation_type=self.params.rnn_r_act,
                normalizer_type=self.params.rnn_r_norm,
                recurrent_cell_type=self.params.rnn_cell_type,
                train=True, hidden_size=self.params.rnn_r_hidden,
                input_depth=input_size, init_data=init_data
            )

        network_shape = [self.params.rnn_r_hidden] + \
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
        self.Network['Linear'] = MLPWithMask(
            dims=network_shape, scope='mlp',
            activation_type=act_type, normalizer_type=norm_type,
            train=True, init_data=init_data, seed=seed
        )

    def __call__(self, input):
        self.Tensor['Intermediate'] = self.Network['Recurrent'](input)[0]
        print(self.Tensor['Intermediate'])
        self.Tensor['Predictions'] = self.Network['Linear'](self.Tensor['Intermediate'])
        print(self.Tensor['Predictions'])
        return self.Tensor['Predictions']

    def weight_variables(self):
        return self.Network['Recurrent'].weights() + self.Network['Linear'].weights()

    def get_mask(self):
        return self.Network['Recurrent'].get_mask() + self.Network['Linear'].get_mask()

    def get_weighted_mask(self):
        return self.Network['Recurrent'].get_weighted_mask() + \
               self.Network['Linear'].get_weighted_mask()




