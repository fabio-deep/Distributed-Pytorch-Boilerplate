# coding: utf-8
import os, warnings, logging, random, argparse
import numpy as np
warnings.filterwarnings("ignore")

import torch
import torchvision
import torch.nn as nn

from resnet import *
from train import *
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=.1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--decay_rate', type=float, default=0.1)
parser.add_argument('--decay_steps', type=int, default=0)
parser.add_argument('--optimiser', default='sgd')
parser.add_argument('--decay_milestones', nargs='+', type=int, default=[0])
parser.add_argument('--padding', type=int, default=4)
parser.add_argument('--brightness', type=float, default=0)
parser.add_argument('--contrast', type=float, default=0)
parser.add_argument('--patience', default=60)
parser.add_argument('--crop_dim', type=int, default=32)
parser.add_argument('--load_checkpoint_dir', default=None)
parser.add_argument('--no_distributed', dest='distributed', action='store_false')
parser.set_defaults(distributed=True)
parser.add_argument('--inference', dest='inference', action='store_true')
parser.set_defaults(inference=False)
parser.add_argument('--half_precision', dest='half_precision', action='store_true')
parser.set_defaults(half_precision=False)

def setup(distributed):
    # kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}') # use this to kill zombie processes
    # run with: python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --use_env main.py
    if distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK'))
        device = torch.device(f'cuda:{local_rank}') # unique on individual node

        print('World size: {} ; Rank: {} ; LocalRank: {} ; Master: {}:{}'.format(
            os.environ.get('WORLD_SIZE'),
            os.environ.get('RANK'),
            os.environ.get('LOCAL_RANK'),
            os.environ.get('MASTER_ADDR'), os.environ.get('MASTER_PORT')))
    else: # run with: python main.py --no_distributed
        local_rank = None
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = 8 # 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # True

    return device, local_rank

def main():
    args = parser.parse_known_args()[0]
    device, local_rank = setup(distributed=args.distributed)

    ''' --------------------------- LOAD DATA -------------------------------'''
    if args.dataset == 'cifar10':
        dataset = 'CIFAR10'
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)
        dataset_paths = {'train': os.path.join(working_dir,'train'),
                         'test':  os.path.join(working_dir,'test')}

        dataloaders = cifar10(args, dataset_paths)

        args.class_names = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # 0,1,2,3,4,5,6,7,8,9 labels
        args.n_channels, args.n_classes = 3, 10

    elif args.dataset == 'svhn':
        dataset = 'SVHN'
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)
        dataset_paths = {'train': os.path.join(working_dir,'train'),
                         #'extra': os.path.join(working_dir,'extra'),
                         'test':  os.path.join(working_dir,'test')}

        dataloaders = svhn(args, dataset_paths)

        args.class_names = ('zero', 'one', 'two', 'three',
            'four', 'five', 'six', 'seven', 'eight', 'nine') # 0,1,2,3,4,5,6,7,8,9 labels
        args.n_channels, args.n_classes = 3, 10

    elif args.dataset == 'fashionmnist':
        dataset = 'FashionMNIST'
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)
        dataset_paths = {'train': os.path.join(working_dir,'train'),
                         'test':  os.path.join(working_dir,'test')}

        dataloaders = fashionmnist(args, dataset_paths)

        args.class_names = ('tshirt', 'trouser', 'pullover', 'dress',
            'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankleboot') # 0,1,2,3,4,5,6,7,8,9 labels
        args.n_channels, args.n_classes = 1, 10

    elif args.dataset == 'mnist':
        dataset = 'MNIST'
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)
        dataset_paths = {'train': os.path.join(working_dir,'train'),
                         'test':  os.path.join(working_dir,'test')}

        dataloaders = mnist(args, dataset_paths)

        args.class_names = ('zero', 'one', 'two', 'three', 'four',
            'five', 'six', 'seven', 'eight', 'nine') # 0,1,2,3,4,5,6,7,8,9 labels
        args.n_channels, args.n_classes = 1, 10

    else:
        NotImplementedError('{} dataset not available.'.format(args.dataset))

    ''''----------------------- EXPERIMENT CONFIG ---------------------------'''
    # check number of models already saved in 'experiments' dir, add 1 to get new model number
    experiments_dir = os.path.join(os.path.split(os.getcwd())[0], 'experiments')
    os.makedirs(experiments_dir, exist_ok=True)
    model_num = len(os.listdir(experiments_dir)) + 1

    # create all save dirs
    model_dir = os.path.join(os.path.split(os.getcwd())[0],
        'experiments', 'Model_'+str(model_num))
    args.summaries_dir = os.path.join(model_dir, 'summaries')
    args.checkpoint_dir = os.path.join(model_dir, 'checkpoint.pt')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(args.summaries_dir, exist_ok=True)

    # save hyperparameters in .txt file
    with open(os.path.join(model_dir, 'hyperparams.txt'), 'w') as logs:
        for key, value in vars(args).items():
            logs.write('--{0}={1} '.format(str(key), str(value)))

    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s',
         handlers=[logging.FileHandler(os.path.join(model_dir, 'trainlogs.txt')),
            logging.StreamHandler()])

    ''' -------------------------- INIT MODEL -------------------------------'''
    model = resnet20(args)

    if args.distributed:
        torch.cuda.set_device(device)
        torch.set_num_threads(1) # n cpu threads / n processes per node
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
            device_ids=[local_rank], output_device=local_rank)
        # only print stuff from process (rank) 0
        args.print_progress = True if int(os.environ.get('RANK')) == 0 else False
    else:
        if args.half_precision:
            model.half()  # convert to half precision
            for layer in model.modules():
                # keep batchnorm in 32 for convergence reasons
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        print('\nUsing', torch.cuda.device_count(), 'GPU(s).\n')
        model.to(device)
        args.print_progress = True

    ''' -------------------------- PRINT INFO -------------------------------'''
    if args.print_progress:
        logging.info('-'*70) # print some info on architecture
        logging.info('{:>25} {:>27} {:>15}'.format('Layer.Parameter', 'Shape', 'Param #'))
        logging.info('-'*70)

        for param in model.state_dict():
            p_name = param.split('.')[-2]+'.'+param.split('.')[-1]
            # don't print batch norm layers for prettyness
            if p_name[:2] != 'BN' and p_name[:2] != 'bn':
                logging.info('{:>25} {:>27} {:>15}'.format(p_name,
                str(list(model.state_dict()[param].squeeze().size())),
                '{0:,}'.format(np.product(list(model.state_dict()[param].size())))))
        logging.info('-'*70)

        logging.info('\nTotal params: {:,}\n\nSummaries dir: {}\n'.format(
            sum(p.numel() for p in model.parameters()),
            args.summaries_dir))

        for key, value in vars(args).items():
            if str(key) != 'print_progress':
                logging.info('--{0}: {1}'.format(str(key), str(value)))

        logging.info('\ntrain: {} - valid: {} - test: {}'.format(
            len(dataloaders['train'].dataset), len(dataloaders['valid'].dataset),
            len(dataloaders['test'].dataset)))

    ''' ---------------------- TRAIN/EVALUATE MODEL -------------------------'''
    if not args.inference:
        score = train(model, dataloaders, args)

        if args.distributed: # cleanup
            torch.distributed.destroy_process_group()
    else:
        model.load_state_dict(torch.load(args.load_checkpoint_dir))
        test_loss, test_acc = evaluate(model, args, dataloaders['test'])
        print('[Test] loss {:.4f} - acc {:.4f} - acc_topk {:.4f}'.format(
            test_loss, test_acc[0], test_acc[1]))

if __name__ == '__main__':
    main()
