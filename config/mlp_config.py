import argparse

def mlp_parser(parser):
    parser.add_argument('--mlp_l_hidden_seq', type=str, default='128,128,64')
    parser.add_argument('--mlp_l_act_seq', type=str, default='relu,relu,relu')
    parser.add_argument('--mlp_l_norm_seq', type=str, default='batch,batch,batch')

    return parser