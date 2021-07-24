import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))

import wandb
from utils.args import *
from algorithm.FetchSGD.server import SERVER as FetchSGD_SERVER
from algorithm.topk.server import SERVER as TopK_SERVER


if __name__ == '__main__':
    wandb.init(entity='tdye24', project='FedPrune')

    args = parse_args()

    # algorithm = args.algorithm
    # dataset_name = args.dataset
    # model_name = args.model
    # num_rounds = args.num_rounds
    # eval_interval = args.eval_interval
    # clients_per_round = args.clients_per_round
    # epoch = args.epoch
    # batch_size = args.batch_size
    # lr = args.lr
    # lr_decay = args.lr_decay
    # decay_step = args.decay_step
    # alpha = args.alpha
    # seed = args.seed
    # cuda = args.cuda

    wandb.watch_called = False
    config = wandb.config
    config.update(args)

    server = None
    if config.algorithm == 'fetchsgd':
        server = FetchSGD_SERVER(config=config)
    elif config.algorithm == 'topk':
        server = TopK_SERVER(config=config)
    server.federate()
    table = PrettyTable(['Average Upload.', 'Average Download.'])
    table.add_row([server.upload_bytes / len(server.clients), server.download_bytes / len(server.clients)])
    print(table)
