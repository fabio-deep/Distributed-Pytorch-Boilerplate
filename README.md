
# Boilerplate Pytorch code with Distributed Training.

This is a Pytorch implementation with general code for running and logging distributed training experiments. 

Simply drop in your own model into `src/main.py`.

  * **Author**: Fabio De Sousa Ribeiro
  * **Email**: fdesosuaribeiro@lincoln.ac.uk

## Run
You can launch **Distributed** training from `src/` using:

    python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --use_env main.py

This will train on a single machine (`nnodes=1`), assigning 1 process per GPU where `nproc_per_node=2` refers to training on 2 GPUs. To train on `N` GPUs simply launch `N` processes by setting `nproc_per_node=N`.

The number of CPU threads to use per process is hard coded to `torch.set_num_threads(1)` for safety, and can be changed to `your # cpu threads / nproc_per_node` for better performance.

For more info on **multi-node** and **multi-gpu** distributed training refer to https://github.com/hgrover/pytorchdistr/blob/master/README.md

To train normally using **nn.DataParallel** or using the CPU:

    python main.py --no_distributed


## Example output when launching an experiment:

```
(torch) 
Documents/Distributed-Pytorch-Boilerplate/src  master ✔                                         43m  ⍉
▶ python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --use_env main.py

World size: 2 ; Rank: 0 ; LocalRank: 0 ; Master: localhost:port
World size: 2 ; Rank: 1 ; LocalRank: 1 ; Master: localhost:port

----------------------------------------------------------------------
          Layer.Parameter                       Shape          Param#
----------------------------------------------------------------------
             conv1.weight               [16, 3, 3, 3]             432
             conv1.weight              [16, 16, 3, 3]           2,304
             conv2.weight              [16, 16, 3, 3]           2,304
             conv1.weight              [16, 16, 3, 3]           2,304
             conv2.weight              [16, 16, 3, 3]           2,304
             conv1.weight              [16, 16, 3, 3]           2,304
             conv2.weight              [16, 16, 3, 3]           2,304
             conv1.weight              [32, 16, 3, 3]           4,608
             conv2.weight              [32, 32, 3, 3]           9,216
          shortcut.weight                    [32, 16]             512
             conv1.weight              [32, 32, 3, 3]           9,216
             conv2.weight              [32, 32, 3, 3]           9,216
             conv1.weight              [32, 32, 3, 3]           9,216
             conv2.weight              [32, 32, 3, 3]           9,216
             conv1.weight              [64, 32, 3, 3]          18,432
             conv2.weight              [64, 64, 3, 3]          36,864
          shortcut.weight                    [64, 32]           2,048
             conv1.weight              [64, 64, 3, 3]          36,864
             conv2.weight              [64, 64, 3, 3]          36,864
             conv1.weight              [64, 64, 3, 3]          36,864
             conv2.weight              [64, 64, 3, 3]          36,864
            linear.weight                    [10, 64]             640
              linear.bias                        [10]              10
----------------------------------------------------------------------

Total params: 272,474

Summaries dir: .../Distributed-Pytorch-Boilerplate/experiments/Model_17/summaries

--dataset: cifar10
--n_epochs: 1000
--batch_size: 128
--learning_rate: 0.1
--weight_decay: 0.0005
--decay_rate: 0.1
--decay_steps: 0
--optimiser: sgd
--decay_milestones: [0]
--padding: 4
--brightness: 0
--contrast: 0
--patience: 60
--crop_dim: 32
--load_checkpoint_dir: None
--distributed: True
--inference: False
--half_precision: False
--class_names: ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
--n_channels: 3
--n_classes: 10
--summaries_dir: .../Distributed-Pytorch-Boilerplate/experiments/Model_17/summaries
--checkpoint_dir: .../Distributed-Pytorch-Boilerplate/experiments/Model_17/checkpoint.pt

train: 45000 - valid: 5000 - test: 10000

Epoch 1/1000:

100%|████████████████████████████████████████████████████████████████| 175/175 [00:05<00:00, 30.29it/s]

[Train] loss: 1.6505 - acc: 0.3918 | [Valid] loss: 1.4670 - acc: 0.4575 - acc_topk: 0.6883

Epoch 2/1000:

100%|████████████████████████████████████████████████████████████████| 175/175 [00:05<00:00, 31.47it/s]

[Train] loss: 1.1477 - acc: 0.5896 | [Valid] loss: 1.0672 - acc: 0.6206 - acc_topk: 0.8073

Epoch 3/1000:

100%|████████████████████████████████████████████████████████████████| 175/175 [00:05<00:00, 31.60it/s]

[Train] loss: 0.9342 - acc: 0.6696 | [Valid] loss: 1.1729 - acc: 0.6176 - acc_topk: 0.7985

Epoch 4/1000:

100%|████████████████████████████████████████████████████████████████| 175/175 [00:05<00:00, 32.01it/s]

[Train] loss: 0.8061 - acc: 0.7194 | [Valid] loss: 0.9524 - acc: 0.6723 - acc_topk: 0.8504
```



