import init_path
from runner.mnist3 import *
from config.base_config import *
from config.mlp_config import *

def main():
    parser = get_base_parser()
    parser = mlp_parser(parser)
    params = make_parser(parser)
    Runner = MNIST3Runner('mnist', params)
    Runner.run()

if __name__ == '__main__':
    main()