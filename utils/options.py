import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--rounds', type=int, default=1, help="total rounds for repeat experiments")
    parser.add_argument('--epochs', type=int, default=300, help="epochs of training")
    parser.add_argument('--num_users', type=int, default=1, help="number of users per domain")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=256, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")

    # dataset argument
    parser.add_argument('--dataset', type=str, default='pacs', help="name of adopted dataset, domainnet|pacs|officehome")
    parser.add_argument('--data_dir', type=str, default='/home/zgg/FL/Data/', help="direction of dataset")

    # model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--local_layers', type=str, default='local_C,local_projector,running_mean,running_var', help='prefix of local layers')

    # training loss
    parser.add_argument('--con_lambda', type=float, default=40.0, help='lambda for supervised contrastive loss')
    parser.add_argument('--con_temp', type=float, default=0.2, help='temperature for supervised contrastive loss')

    # training argument
    parser.add_argument('--mode', type=str, default='train', help="working mode: train or test")
    parser.add_argument('--model_step', type=int, default=301, help="step for saving model")
    parser.add_argument('--best_model', action='store_true', help="whether saving the best model or not")
    parser.add_argument('--test_step', type=int, default=1, help="step for model testing")
    parser.add_argument('--log_step', type=int, default=1, help="step for print training log")
    parser.add_argument('--save_dir', type=str, default='', help="record direction")

    # other arguments
    parser.add_argument('--wandb_mode', type=str, default='disabled', help='mode for wandb, online|offline|disabled')
    parser.add_argument('--wandb_key', type=str, default='', help='API key for wandb login')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', type=int, default=1, help="verbose print")
    parser.add_argument('--fix_seed', type=int, default=1, help="whether fixes random seed or not (1: fixed)")
    parser.add_argument('--seed', nargs='+', default=[0, 1, 2, 3, 4], help="random seed")

    args = parser.parse_args()
    return args
