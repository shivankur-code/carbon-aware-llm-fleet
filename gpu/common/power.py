"""GPU power sampling via NVML (nvidia-ml-py), with an nvidia-smi fallback.

    with PowerSampler(hz=10) as ps:
        run_workload()
    ps.energy_j, ps.mean_w, ps.samples

Energy is integrated (trapezoid) from per-GPU board power. Host (CPU/DRAM/fan)
power is NOT visible to NVML; the ingest step adds a host-power model.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time


def _nvml():
    try:
        import pynvml  # provided by the nvidia-ml-py package
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def gpu_inventory() -> list[dict]:
    nv = _nvml()
    out = []
    if nv is not None:
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            name = nv.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
            mem = nv.nvmlDeviceGetMemoryInfo(h).total / 2**30
            try:
                limit = nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000
            except Exception:
                limit = None
            out.append(dict(index=i, name=name, memory_gib=round(mem, 1), power_limit_w=limit))
        try:
            drv = nv.nvmlSystemGetDriverVersion()
            for g in out:
                g["driver"] = drv.decode() if isinstance(drv, bytes) else drv
        except Exception:
            pass
    elif shutil.which("nvidia-smi"):
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,power.limit,driver_version",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        for line in q.strip().splitlines():
            i, name, mem, lim, drv = [x.strip() for x in line.split(",")]
            out.append(dict(index=int(i), name=name, memory_gib=round(float(mem) / 1024, 1),
                            power_limit_w=float(lim) if lim not in ("[N/A]", "") else None, driver=drv))
    return out


class PowerSampler:
    def __init__(self, hz: float = 10.0, gpus: list[int] | None = None):
        self.period = 1.0 / hz
        self.gpus = gpus
        self.samples: list[tuple[float, float]] = []  # (t, total watts over selected GPUs)
        self._stop = threading.Event()
        self._nv = _nvml()
        self.available = self._nv is not None or shutil.which("nvidia-smi") is not None

    def _read(self) -> float | None:
        nv = self._nv
        if nv is not None:
            idx = self.gpus if self.gpus is not None else range(nv.nvmlDeviceGetCount())
            return sum(nv.nvmlDeviceGetPowerUsage(nv.nvmlDeviceGetHandleByIndex(i)) for i in idx) / 1000.0
        if shutil.which("nvidia-smi"):
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,power.draw", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout
            tot = 0.0
            for line in q.strip().splitlines():
                i, w = [x.strip() for x in line.split(",")]
                if self.gpus is None or int(i) in self.gpus:
                    tot += float(w)
            return tot
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            w = self._read()
            if w is not None:
                self.samples.append((time.time(), w))
            self._stop.wait(self.period)

    def __enter__(self) -> "PowerSampler":
        self.t0 = time.time()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._th.join()
        self.t1 = time.time()

    @property
    def seconds(self) -> float:
        return self.t1 - self.t0

    @property
    def energy_j(self) -> float | None:
        if len(self.samples) < 2:
            return None
        e = 0.0
        for (ta, wa), (tb, wb) in zip(self.samples, self.samples[1:]):
            e += 0.5 * (wa + wb) * (tb - ta)
        return e

    @property
    def mean_w(self) -> float | None:
        return sum(w for _, w in self.samples) / len(self.samples) if self.samples else None
