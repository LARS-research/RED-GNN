import numpy as np
from scipy.stats import rankdata
import subprocess
import logging

def cal_ranks(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    full_rank = rankdata(-scores, method='average', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='min', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels      # get the ranks of multiple answering entities simultaneously
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)


def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, h_1, h_10


def select_gpu():
    nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
    gpu_info = False
    gpu_info_line = 0
    proc_info = False
    gpu_mem = []
    gpu_occupied = set()
    i = 0
    for line in nvidia_info.stdout.split(b'\n'):
        line = line.decode().strip()
        if gpu_info:
            gpu_info_line += 1
            if line == '':
                gpu_info = False
                continue
            if gpu_info_line % 3 == 2:
                mem_info = line.split('|')[2]
                used_mem_mb = int(mem_info.strip().split()[0][:-3])
                gpu_mem.append(used_mem_mb)
        if proc_info:
            if line == '|  No running processes found                                                 |':
                continue
            if line == '+-----------------------------------------------------------------------------+':
                proc_info = False
                continue
            proc_gpu = int(line.split()[1])
            #proc_type = line.split()[3]
            gpu_occupied.add(proc_gpu)
        if line == '|===============================+======================+======================|':
            gpu_info = True
        if line == '|=============================================================================|':
            proc_info = True
        i += 1
    for i in range(0,len(gpu_mem)):
        if i not in gpu_occupied:
            logging.info('Automatically selected GPU %d because it is vacant.', i)
            return i
    for i in range(0,len(gpu_mem)):
        if gpu_mem[i] == min(gpu_mem):
            logging.info('All GPUs are occupied. Automatically selected GPU %d because it has the most free memory.', i)
            return i

# ---------------- Memory tracking utilities ----------------
# Tracks peak CPU RSS and (optionally) CUDA peak memory during a code region.
# Usage:
#   meter = PeakMemoryMeter(track_cuda=True)
#   meter.reset()
#   ... do work ...
#   meter.update()  # call periodically
#   summary = meter.summary_mb()

import os
import time

def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)

def _get_cpu_rss_bytes() -> int:
    """Return current process RSS in bytes."""
    # Prefer psutil for accuracy; fall back to resource.getrusage.
    try:
        import psutil  # type: ignore
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        try:
            import resource
            # ru_maxrss is KB on Linux, bytes on macOS. We normalize to bytes.
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Heuristic: if value is huge, it's probably bytes already.
            if ru > 10**10:
                return int(ru)
            # Assume KB (Linux).
            return int(ru) * 1024
        except Exception:
            return 0

def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

def _reset_cuda_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass

def _get_cuda_peak_bytes() -> dict:
    """Return CUDA peak memory stats (bytes) for current device."""
    out = {"allocated": 0, "reserved": 0}
    try:
        import torch
        if torch.cuda.is_available():
            out["allocated"] = int(torch.cuda.max_memory_allocated())
            out["reserved"] = int(torch.cuda.max_memory_reserved())
    except Exception:
        pass
    return out

class PeakMemoryMeter:
    def __init__(self, track_cpu: bool = True, track_cuda: bool = True):
        self.track_cpu = track_cpu
        self.track_cuda = track_cuda
        self.reset()

    def reset(self):
        self._cpu_peak = 0
        self._cuda_peak_alloc = 0
        self._cuda_peak_reserved = 0
        self._t0 = time.time()
        if self.track_cuda and _cuda_available():
            _reset_cuda_peak()
        # initialize with current values
        self.update()

    def update(self):
        if self.track_cpu:
            rss = _get_cpu_rss_bytes()
            if rss > self._cpu_peak:
                self._cpu_peak = rss
        if self.track_cuda and _cuda_available():
            stats = _get_cuda_peak_bytes()
            self._cuda_peak_alloc = max(self._cuda_peak_alloc, stats["allocated"])
            self._cuda_peak_reserved = max(self._cuda_peak_reserved, stats["reserved"])

    def summary_mb(self) -> dict:
        return {
            "cpu_rss_peak_mb": _bytes_to_mb(self._cpu_peak),
            "cuda_alloc_peak_mb": _bytes_to_mb(self._cuda_peak_alloc),
            "cuda_reserved_peak_mb": _bytes_to_mb(self._cuda_peak_reserved),
            "elapsed_s": time.time() - self._t0,
        }
