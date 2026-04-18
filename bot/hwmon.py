"""Lightweight hardware monitor — samples GPU VRAM, GPU util, and RAM while
a block of work runs, then reports peak and average usage.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time

_NOWWIN = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
from dataclasses import dataclass, field


@dataclass
class HWStats:
    label: str
    duration_s: float = 0.0
    samples: int = 0
    ram_used_gb_peak: float = 0.0
    gpu_vram_gb_peak: float = 0.0
    gpu_util_peak: int = 0
    gpu_temp_peak: int = 0
    gpu_power_peak: float = 0.0
    gpu_vram_gb_mean: float = 0.0
    gpu_util_mean: float = 0.0
    vram_samples: list[float] = field(default_factory=list)
    util_samples: list[int] = field(default_factory=list)

    def report(self) -> str:
        return (
            f"[hwmon] {self.label}: {self.duration_s:.1f}s | "
            f"GPU VRAM peak {self.gpu_vram_gb_peak:.2f} GB "
            f"(avg {self.gpu_vram_gb_mean:.2f}) | "
            f"GPU util peak {self.gpu_util_peak}% "
            f"(avg {self.gpu_util_mean:.0f}%) | "
            f"GPU {self.gpu_temp_peak}C, {self.gpu_power_peak:.0f}W | "
            f"RAM peak {self.ram_used_gb_peak:.1f} GB"
        )


def _query_nvidia_smi() -> tuple[float, int, int, float] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True, timeout=2,
            creationflags=_NOWWIN,
        )
        line = out.strip().splitlines()[0]
        mem, util, temp, power = [p.strip() for p in line.split(",")]
        return (float(mem) / 1024.0, int(util), int(temp), float(power))
    except Exception:
        return None


def _ram_used_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().used / 1e9
    except Exception:
        return 0.0


class Monitor:
    """Context manager — samples hardware every `interval` seconds."""

    def __init__(self, label: str, interval: float = 0.5) -> None:
        self.stats = HWStats(label=label)
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def __enter__(self) -> "Monitor":
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.stats.duration_s = time.time() - self._t0
        if self.stats.vram_samples:
            self.stats.gpu_vram_gb_mean = sum(self.stats.vram_samples) / len(self.stats.vram_samples)
            self.stats.gpu_util_mean = sum(self.stats.util_samples) / len(self.stats.util_samples)
        print(self.stats.report())

    def _loop(self) -> None:
        while not self._stop.is_set():
            gpu = _query_nvidia_smi()
            ram = _ram_used_gb()
            self.stats.samples += 1
            self.stats.ram_used_gb_peak = max(self.stats.ram_used_gb_peak, ram)
            if gpu is not None:
                vram, util, temp, power = gpu
                self.stats.vram_samples.append(vram)
                self.stats.util_samples.append(util)
                self.stats.gpu_vram_gb_peak = max(self.stats.gpu_vram_gb_peak, vram)
                self.stats.gpu_util_peak = max(self.stats.gpu_util_peak, util)
                self.stats.gpu_temp_peak = max(self.stats.gpu_temp_peak, temp)
                self.stats.gpu_power_peak = max(self.stats.gpu_power_peak, power)
            if self._stop.wait(self.interval):
                break
