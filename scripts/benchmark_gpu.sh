#!/bin/bash
# Usage: ./scripts/benchmark_gpu.sh [category] [n_runs]

set -e

CATEGORY=${1:-screw}
N_RUNS=${2:-100}
WEIGHTS="models/checkpoints/${CATEGORY}_patchcore.pt"

echo "========================================"
echo " Robot Vision — GPU Benchmark"
echo " Category : $CATEGORY"
echo " Runs     : $N_RUNS"
echo "========================================"

python -c "
import torch, time
from src.anomaly.patchcore import PatchCore
from src.gpu.gpu_utils import get_device, get_memory_stats
from PIL import Image
import numpy as np

device = get_device()
get_memory_stats()

model = PatchCore(device=device)
model.load('$WEIGHTS')

# warm-up
dummy = torch.randn(1, 3, 224, 224)
for _ in range(10):
    model.score(dummy)

# benchmark
times = []
for i in range($N_RUNS):
    dummy = torch.randn(1, 3, 224, 224)
    start = time.perf_counter()
    model.score(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)

times = sorted(times)
print(f'\n--- Benchmark Results ({$N_RUNS} runs) ---')
print(f'Mean latency  : {sum(times)/len(times):.2f} ms')
print(f'Median latency: {times[len(times)//2]:.2f} ms')
print(f'P95 latency   : {times[int(len(times)*0.95)]:.2f} ms')
print(f'Min / Max     : {min(times):.2f} ms / {max(times):.2f} ms')
print(f'Throughput    : {1000/(sum(times)/len(times)):.1f} FPS')

get_memory_stats()
"

echo ""
echo "[DONE] Benchmark complete."
