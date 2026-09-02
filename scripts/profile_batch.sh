#!/bin/bash
#SBATCH --job-name=rangedit-prof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --nodelist=cn07
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-prof-%j.out
#SBATCH --error=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-prof-%j.out
set -euo pipefail
cd /csehome/p24cs0005/rangediffseg
module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh
source /csehome/p24cs0005/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}
conda activate difforecast
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u - <<'PYEOF'
import torch, time, sys
sys.path.insert(0,'/csehome/p24cs0005/rangediffseg')
import models
dev='cuda'
tot=torch.cuda.get_device_properties(0).total_memory/2**30
print(f'device: {torch.cuda.get_device_name(0)}  total: {tot:.1f} GiB')
print(f'torch {torch.__version__}  arch_list={torch.cuda.get_arch_list()}\n')

m = models.RangeDiT(in_channels=5,n_cls=20,backbone='DiT-XL/2',image_size=(64,384),
      pretrained_path=None,new_patch_size=(2,8),new_patch_stride=(2,8),skip_filters=256,
      decoder='up_conv',up_conv_d_decoder=256,up_conv_scale_factor=(2,8),
      fusion_layers=[5,11,17,23],adapter_layers=[3,9,15,21,27]).to(dev)
m.freeze_encoder(unfreeze_adaln=True,adaln_bias_only=True)
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-4)
scaler=torch.amp.GradScaler('cuda')
m.train()
print(f'{"batch":>6}{"peak GiB":>11}{"s/iter":>9}{"scans/s":>10}')
for bs in [4,8,16,24,32,48]:
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x=torch.randn(bs,5,64,384,device=dev); y=torch.randint(0,20,(bs,64,384),device=dev)
        for it in range(4):
            if it==1: torch.cuda.synchronize(); t0=time.time()
            with torch.amp.autocast('cuda'):
                out=m(x); loss=torch.nn.functional.cross_entropy(out,y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize(); dt=(time.time()-t0)/3
        pk=torch.cuda.max_memory_allocated()/2**30
        print(f'{bs:>6}{pk:>10.1f}G{dt:>9.3f}{bs/dt:>10.1f}')
        del x,y,out,loss
    except torch.cuda.OutOfMemoryError:
        print(f'{bs:>6}{"OOM":>11}'); torch.cuda.empty_cache(); break
PYEOF
