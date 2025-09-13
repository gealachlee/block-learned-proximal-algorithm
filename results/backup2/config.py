import configargparse
import os
import torch
import utils

# Argument Parsing
parser = configargparse.get_arg_parser(description='Configurations for block-lista')

# Problem parameters
parser.add('--tau', type=float, default=0.1, metavar='FLOAT',
           help='Parameter for reg. term in the objective function')  # 0.1

# Data parameters
parser.add('--m', type=int, default=256, metavar='INT',
           help='Number of rows in matrix A')
parser.add('--n', type=int, default=512, metavar='INT',
           help='Number of cols in matrix A')
parser.add('--sparsity', type=int, default=8, metavar='INT')
parser.add('--gLen', type=int, default=8, metavar='INT')
parser.add('--x-mu', type=float, default=0.0, metavar='FLOAT',
           help='For training: mean of the Gaussian distribution of nonzero entries')
parser.add('--x-sigma', type=float, default=1.0, metavar='FLOAT',
           help='For training: std dev. of the Gaussian distribution of nonzero entries')
parser.add('--snr', type=float, default=1e10, metavar='FLOAT',
           help='SNR for additive Gaussian noise')
parser.add('--data-seed', type=int, default=1, metavar='INT',
           help='Random seed for the data generation')

# Model parameters
parser.add('--layers', type=int, default=10, metavar='INT',
           help='Number of layers of the neural network')

# Training parameters
parser.add('--objective', type=str, default='GT', metavar='{OBJECTIVE,L2,L1,GT}',
           help='Objective used for the training')
parser.add('--save-dir', type=str, default='temp',
           help='Saving directory for saved models and logs')
parser.add('--train-size', type=int, default=12800, metavar='N',  # 51200 12800
           help='Number of training samples')
parser.add('--val-size', type=int, default=2048, metavar='N',
           help='Number of validation samples')
parser.add('--test-size', type=int, default=2048, metavar='N',
           help='Number of testing samples')
parser.add('--train-batch-size', type=int, default=512, metavar='N',  # 512, metavar='N',
           help='Batch size for training')
parser.add('--val-batch-size', type=int, default=2048, metavar='N',  # 2048
           help='Batch size for validation')
parser.add('--test-batch-size', type=int, default=2048, metavar='N',  # 2048
           help='Batch size for testing')
parser.add('--init-lr', type=float, default=0.01, metavar='FLOAT',  # default=0.1, metavar='FLOAT',
           help='Initial learning rate')
parser.add('--lr-decay-layer', type=float, default=0.3, metavar='FLOAT',
           help='Decay learning rates of trained layers')
parser.add('--lr-decay-stage2', type=float, default=0.2,
           metavar='FLOAT', help='Decay rate for training stage2 in each layer')
parser.add('--lr-decay-stage3', type=float, default=0.02, metavar='FLOAT',
           help='Decay rate for training stage3 in each layer')
parser.add('--best-wait', type=int, default=5, metavar='N',
           help='Wait time for better validation performance')

opts, _ = parser.parse_known_args()

# Save directory
opts.save_dir = os.path.join('results', opts.save_dir)
if not os.path.isdir(opts.save_dir):
    os.makedirs(opts.save_dir)

# Logging file
logger_file = os.path.join(opts.save_dir, 'train.log')
opts.logger = utils.setup_logger(logger_file)
opts.logger('Checkpoints will be saved to directory `{}`'.format(opts.save_dir))
opts.logger('Log file for training will be saved to file `{}`'.format(logger_file))

# Use cuda if it is available
opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'
opts.dtype = torch.float64
opts.logger('Using device: {}'.format(opts.device))  # Output the type of device used
opts.logger('Using tau: {}'.format(opts.tau))  # Output the tau used in current exp

