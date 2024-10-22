import utils.random_state
import argparse
from utils.options import args_parser
import config
from src.server import Server
import time
import os
import wandb
import random
import gc

# print parameters of experiments
def exp_parameter(args):
    print(f'Communication Rounds: {args.epochs}')
    print(f'Client Number for Each Domain: {args.num_users}')
    print(f'Local Epochs: {args.local_ep}')
    print(f'Local Batch Size: {args.local_bs}')
    print(f'Learning Rate: {args.lr}')
    print(f'Model Type: {args.model}')
    print(f'Domains: {args.domains}')

def set_config():
    args = args_parser()
    dataset_config = getattr(config, args.dataset)

    args_dict = args.__dict__
    args_dict['data_dir'] = '{}{}'.format(args.data_dir, dataset_config['data_dir'])
    # args_dict.update(dataset_config)
    for key in dataset_config.keys():
        if key in args_dict.keys():
            continue
        else:
            args_dict[key] = dataset_config[key]

    now_time = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))

    base_str = f'DualFed_{args.dataset}_{args.local_layers}_{args.local_bs}_{args.local_ep}_{args.train_num}_{args.num_users}_{args.model}' \
               f'_con_temp{args.con_temp}_con_lambda_{args.con_lambda}'
    args_dict['save_dir'] = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "."))}/save/{now_time}_' \
                             f'{base_str}_{random.randint(100, 999)}/'

    args_dict['wandb_name'] = f'{base_str}'

    args = argparse.Namespace(**args_dict)
    return args

def train(args):
    total_rounds = args.rounds
    for round in range(total_rounds):
        args.__setattr__('current_round', round)

        # set random seed
        if args.fix_seed == 1:
            if round < len(args.seed):
                utils.random_state.setRandomState(args.seed[round])
            else:
                utils.random_state.setRandomState(args.seed[-1])

        # initialize wandb
        run = wandb.init(project='DualFed', config=args, dir=os.getcwd())
        wandb.run.name = f'{args.wandb_name}_{wandb.run.id}'

        server = Server(args, round)
        server.train()

        run.finish()
        del server
        gc.collect()

if __name__ == '__main__':
    # get experiment configuration
    args = set_config()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # set wandb
    os.environ['WANDB_API_KEY'] = args.wandb_key
    if args.wandb_mode == 'online' and args.wandb_key != '':
        wandb.login()
    else:
        os.environ['WANDB_MODE'] = args.wandb_mode

    # print experiment parameters
    exp_parameter(args)

    # run experiment
    train(args)



