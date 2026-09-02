"""
Minimal distributed-training helpers (single node, one process per GPU).

Launched with torchrun, which sets RANK / LOCAL_RANK / WORLD_SIZE. When those
are absent everything degrades to single-process behaviour, so the same script
runs unchanged on one GPU.
"""

import os

import torch
import torch.distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank():
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0


def get_world_size():
    return dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1


def is_main_process():
    return get_rank() == 0


def setup():
    """Initialise the process group from the torchrun environment."""
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size <= 1:
        return 0, 1, 0

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    # Bind the rank to its device explicitly: without this NCCL infers the
    # mapping from the global rank, which it warns can hang when the
    # rank-to-GPU mapping is not uniform.
    dist.init_process_group(backend='nccl', init_method='env://',
                            device_id=torch.device(f'cuda:{local_rank}'))
    return dist.get_rank(), dist.get_world_size(), local_rank


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier():
    if is_distributed():
        dist.barrier()


def all_reduce_sum(tensor):
    """Sum a tensor across ranks. Returns the input unchanged when single-process."""
    if not is_distributed():
        return tensor
    on_cuda = tensor.is_cuda
    work = tensor if on_cuda else tensor.cuda()
    dist.all_reduce(work, op=dist.ReduceOp.SUM)
    return work if on_cuda else work.cpu()
