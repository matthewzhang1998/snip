import init_path
from config.base_config import *
from config.rnn_config import *
import runner.rnn_ptb4
import runner.rnn_ptb6

def main():
    import logging
    import tensorflow

    parser = get_base_parser()
    parser = rnn_parser(parser)
    params = make_parser(parser)
    if params.exp == 'ptb4':
        Runner = runner.rnn_ptb4.PTBRunner('mnist', params)
    elif params.exp == 'ptb6':
        Runner = runner.rnn_ptb6.PTBRunner('mnist', params)
    Runner.run()

if __name__ == '__main__':
    main()