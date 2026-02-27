import numpy as np
from scipy.stats import rankdata
import subprocess
import logging
import json

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


# -----------------------------
# Peak memory monitoring helpers
# -----------------------------
import os
import threading
import time

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

def _format_bytes(num_bytes: int) -> str:
    if num_bytes is None:
        return "N/A"
    num = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024.0 or unit == "TB":
            return f"{num:.2f}{unit}"
        num /= 1024.0
    return f"{num_bytes}B"

class PeakRSSMonitor:
    """
    Lightweight process RSS peak monitor via periodic sampling.
    This provides per-section peak RSS, unlike ru_maxrss which is process-lifetime.
    """
    def __init__(self, interval_sec: float = 0.1):
        self.interval_sec = float(interval_sec)
        self._stop = threading.Event()
        self._thread = None
        self.peak_rss_bytes = 0
        self._pid = os.getpid()

    def _run(self):
        if psutil is None:
            return
        proc = psutil.Process(self._pid)
        peak = 0
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > peak:
                    peak = rss
            except Exception:
                pass
            time.sleep(self.interval_sec)
        self.peak_rss_bytes = max(self.peak_rss_bytes, peak)

    def start(self):
        self.peak_rss_bytes = 0
        self._stop.clear()
        if psutil is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

def get_cuda_peak_memory_bytes() -> dict:
    """
    Return CUDA peak memory stats in bytes (allocated/reserved).
    Call torch.cuda.reset_peak_memory_stats() before the measured section.
    """
    try:
        import torch  # local import to avoid hard dependency
        if not torch.cuda.is_available():
            return {"alloc": 0, "reserved": 0}
        return {
            "alloc": int(torch.cuda.max_memory_allocated()),
            "reserved": int(torch.cuda.max_memory_reserved()),
        }
    except Exception:
        return {"alloc": 0, "reserved": 0}

def write_memory_report(path: str, tag: str, cuda_stats: dict, rss_peak_bytes: int):
    """
    Append a single-line memory report (human readable + JSON) to `path`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = (
        f"[{tag}] "
        f"GPU_peak_alloc={_format_bytes(cuda_stats.get('alloc', 0))}, "
        f"GPU_peak_reserved={_format_bytes(cuda_stats.get('reserved', 0))}, "
        f"CPU_peak_RSS={_format_bytes(rss_peak_bytes)} "
        f"| json={json.dumps({'tag': tag, 'gpu_alloc_bytes': cuda_stats.get('alloc', 0), 'gpu_reserved_bytes': cuda_stats.get('reserved', 0), 'cpu_rss_peak_bytes': rss_peak_bytes})}\n"
    )
    with open(path, "a+", encoding="utf-8") as f:
        f.write(line)
    print(line.strip())
