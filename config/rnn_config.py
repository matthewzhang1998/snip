import argparse

def rnn_parser(parser):
    parser.add_argument('--rnn_cell_type', type=str, default='lstm')

    parser.add_argument('--rnn_r_hidden_seq', type=str, default='20')
    parser.add_argument('--rnn_r_act_seq', type=str, default='none,none')
    parser.add_argument('--rnn_r_norm_seq', type=str, default='none,none')
    parser.add_argument('--rnn_l_hidden_seq', type=str, default=None)
    parser.add_argument('--rnn_l_act_seq', type=str, default='none')
    parser.add_argument('--rnn_l_norm_seq', type=str, default='none')

    parser.add_argument('--rnn_bidirectional', type=int, default=0)
    parser.add_argument('--rnn_dilated', type=int, default=0)

    return parser