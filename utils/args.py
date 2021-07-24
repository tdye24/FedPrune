import argparse

ALGORITHMS = ['fedavg', 'topk', 'fetchsgd', 'fedprune']
DATASETS = ['cifar10', 'cifar10_iid', 'mnist', 'cifar100']
ERROR_TYPES = ['none', 'local', 'virtual']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm',
                        help='algorithm',
                        choices=ALGORITHMS,
                        required=True)

    parser.add_argument('--dataset',
                        help='name of dataset',
                        choices=DATASETS,
                        required=True)

    parser.add_argument('--model',
                        help='name of model',
                        type=str,
                        required=True)

    parser.add_argument('--numRounds',
                        help='# of communication round',
                        type=int,
                        default=100)

    parser.add_argument('--evalInterval',
                        help='communication rounds between two evaluation',
                        type=int,
                        default=1)

    parser.add_argument('--clientsPerRound',
                        help='# of selected clients per round',
                        type=int,
                        default=1)

    parser.add_argument('--epoch',
                        help='# of epochs when clients train on data',
                        type=int,
                        default=1)

    parser.add_argument('--batchSize',
                        help='batch size when clients train on data',
                        type=int,
                        default=1)

    parser.add_argument('--lr',
                        help='learning rate for local optimizers',
                        type=float,
                        default=3e-4)

    parser.add_argument('--lrDecay',
                        help='decay rate for learning rate',
                        type=float,
                        default=0.99)

    parser.add_argument('--decayStep',
                        help='decay rate for learning rate',
                        type=int,
                        default=200)

    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting',
                        type=int,
                        default=24)

    parser.add_argument('--cuda',
                        help='using cuda',
                        type=bool,
                        default=True)

    parser.add_argument('--numRows',
                        help='num of rows for sketch',
                        type=int,
                        default=5)

    parser.add_argument('--numColumns',
                        help='num of columns for sketch',
                        type=int,
                        default=100000)

    parser.add_argument('--k',
                        type=int,
                        default=50000)

    parser.add_argument('--topkDown',
                        type=bool,
                        default=False)

    parser.add_argument('--localMomentum',
                        type=float,
                        default=0.0)

    parser.add_argument('--globalMomentum',
                        type=float,
                        default=0.9)

    parser.add_argument('--weightDecay',
                        type=float,
                        default=5e-4)

    parser.add_argument('--errorType',
                        choices=ERROR_TYPES,
                        default='none')

    parser.add_argument('--max_grad_norm',
                        help='Clipping gradient norm, is per-worker',
                        type=float)

    return parser.parse_args()
