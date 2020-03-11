
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

