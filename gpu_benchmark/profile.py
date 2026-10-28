"""Cactus smoke/full orchestration. All results remain outside source control."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('profile',choices=['smoke','full'])
    p.add_argument('--output',required=True);p.add_argument('--data',default='data/imagenette2-320')
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    python=sys.executable;start=time.monotonic();deadline=start+(28800 if a.profile=='full' else 7200)
    def call(arguments, logfile):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Profile budget exhausted')
        from .run import execute
        result=execute([python]+arguments,os.environ.copy(),remaining,out/logfile)
        if result:raise RuntimeError(f'{logfile}: exit {result}; inspect log')
    smoke=a.profile=='smoke'
    # Hardware first establishes single-device and interconnect correctness.
    for gpu in range(4):
        # Each process selects its device via LOCAL_RANK without creating a process group.
        env=os.environ.copy();env['LOCAL_RANK']=str(gpu)
        from .run import execute
        code=execute([python,'-m','gpu_benchmark.hardware','--sizes',*(['256'] if smoke else ['1024','4096','8192']),
            '--steps','3' if smoke else '100','--warmup','1' if smoke else '10','--output',str(out/f'hardware-gpu{gpu}.json')],env,min(900,deadline-time.monotonic()),out/f'hardware-gpu{gpu}.log')
        if code:raise RuntimeError(f'Hardware GPU {gpu} failed')
    for count in [2,3,4]:
        call(['-m','torch.distributed.run','--standalone',f'--nproc_per_node={count}','--module','gpu_benchmark.hardware','--communication',
              '--steps','3' if smoke else '100','--output',str(out/f'nccl-{count}.json')],f'nccl-{count}.log')
    common=['-m','gpu_benchmark.run','--gpu-counts','1','2','3','4','--output-dir',str(out/'training'),
        '--precisions','fp32','tf32','fp16','bf16','fp8','bf16-compiled',
        '--budget-seconds',str(max(1,int(deadline-time.monotonic())))]
    if smoke:
        common += ['--models','vit_base_patch16_224','--steps','2','--warmup-steps','1','--validation-steps','1','--repetitions','1',
                   '--vision_batch_size','2','--language_batch_size','2','--synthetic','--n_workers','0','--case-timeout','900']
    else:
        common += ['--vision_data',a.data,'--steps','100','--case-timeout','900']
    call(common,'training.log')
    summary=[json.loads(path.read_text()) for path in (out/'training').glob('*/result.json')]
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(f'Profile complete: {out}',flush=True)
if __name__=='__main__':main()
