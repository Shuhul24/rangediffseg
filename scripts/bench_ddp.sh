#!/bin/bash
#SBATCH --job-name=rangedit-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --nodelist=cn07
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-bench-%j.out
#SBATCH --error=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-bench-%j.out
set -euo pipefail
cd /csehome/p24cs0005/rangediffseg
module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh
source /csehome/p24cs0005/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}
conda activate difforecast
export OMP_NUM_THREADS=8 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000+RANDOM%20000))
torchrun --standalone --nproc_per_node=2 - <<'PYEOF'
import os, time, torch, torch.nn as nn, sys
sys.path.insert(0,'/csehome/p24cs0005/rangediffseg')
import models, utils.dist as D

rank, world, local = D.setup()
dev = torch.device(f'cuda:{local}')
BS = 16

def build(sync_bn):
    m = models.RangeDiT(in_channels=5,n_cls=20,backbone='DiT-XL/2',image_size=(64,384),
        pretrained_path=None,new_patch_size=(2,8),new_patch_stride=(2,8),skip_filters=256,
        decoder='up_conv',up_conv_d_decoder=256,up_conv_scale_factor=(2,8),
        fusion_layers=[5,11,17,23],adapter_layers=[3,9,15,21,27]).to(dev)
    m.freeze_encoder(unfreeze_adaln=True, adaln_bias_only=True)
    if sync_bn:
        m = nn.SyncBatchNorm.convert_sync_batchnorm(m)
    return nn.parallel.DistributedDataParallel(
        m, device_ids=[local], output_device=local, find_unused_parameters=False)

def bench(sync_bn, iters=25):
    m = build(sync_bn)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    scaler = torch.amp.GradScaler('cuda')
    x = torch.randn(BS,5,64,384,device=dev); y = torch.randint(0,20,(BS,64,384),device=dev)
    m.train()
    for i in range(iters):
        if i == 5:
            torch.cuda.synchronize(); D.barrier(); t0 = time.time()
        with torch.amp.autocast('cuda'):
            loss = nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize(); D.barrier()
    dt = (time.time()-t0)/(iters-5)
    del m, opt; torch.cuda.empty_cache()
    return dt

if rank == 0:
    print(f'\n=== DDP benchmark: {world} GPUs, batch {BS}/GPU = {BS*world} effective ===')
    print('(synthetic tensors: isolates compute+comms, excludes data loading)\n')
for sync in (False, True):
    dt = bench(sync)
    if rank == 0:
        print(f'sync_bn={str(sync):<5}  {dt:.3f} s/iter   {BS*world/dt:6.1f} scans/s')
if rank == 0:
    print('\nreference (measured earlier, 1 GPU synthetic): batch 16 = 0.644 s/iter = 24.8 scans/s')
    print('ideal 2-GPU scaling would put 32 samples at ~0.65 s/iter\n')
D.cleanup()
PYEOF
