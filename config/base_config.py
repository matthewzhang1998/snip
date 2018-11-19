import argparse

def get_base_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--val_steps', type=int, default=50)

    parser.add_argument('--optimizer_type', type=str, default='adam')

    parser.add_argument('--noise_delta', type=float, default=1e-2)
    parser.add_argument('--prune_k', type=float, default=0.95)
    parser.add_argument('--random_k', type=float, default=0.95)

    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--model_type', type=str, default='mlp')

    return parser

def make_parser(parser):
    return post_process(parser.parse_args())

def post_process(args):
    # parse the network shape
    for key in dir(args):
        if 'seq' in key:
            if 'hidden' in key:
                setattr(args, key, [int(dim) for dim in getattr(args, key).split(',')])
            else:
                setattr(args, key, [str(dim) for dim in getattr(args, key).split(',')])
    return args


