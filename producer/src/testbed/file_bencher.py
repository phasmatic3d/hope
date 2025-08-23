import math
import time

import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from draco_wrapper import DracoWrapper


@dataclass(slots=True)
class Metrics:
    file: str = ""
    points: int = 0
    raw_bytes: int = 0
    encoded_bytes: int = 0
    compression_ms_mean: float = 0.0
    compression_ms_std: float = 0.0
    ratio_raw_over_encoded: Optional[float] = 0.0
    bits_per_point: Optional[float] = 0.0

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self):
        return iter(self.__dataclass_fields__.keys())

    def __len__(self):
        return len(self.__dataclass_fields__)

    def as_row(self) -> dict:
        return {
            "file": self.file,
            "points": self.points,
            "raw_bytes": self.raw_bytes,
            "encoded_bytes": self.encoded_bytes,
            "compression_ms_mean": self.compression_ms_mean,
            "compression_ms_std": self.compression_ms_std,
            "ratio_raw_over_encoded": self.ratio_raw_over_encoded,
            "bits_per_point": self.bits_per_point,
        }

    @staticmethod
    def csv_header() -> List[str]:
        return list(Metrics().as_row().keys())



class FileBencher:

    def __init__(self, warmup: bool = True, drop_first: bool = True):
        self.warmup = warmup
        self.drop_first = drop_first

    @staticmethod
    def _safe_ratio(numer: float, denom: float) -> float:
        return (numer / denom) if denom > 0 else math.nan

    def bench_file(self, path: Path, wrapper: DracoWrapper, repeats: int = 3) -> Metrics:

        try:
            points, colors = DracoWrapper.load_ply_points_colors(path)
        except Exception as e:
            return Metrics(file=str(path)) 

        if points.size == 0:
            return Metrics()


        if(self.warmup):
            _ = wrapper.encode(points, colors)

        times = []
        encoded_sizes = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            buf = wrapper.encode(points, colors)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            encoded_sizes.append(len(buf))

        if self.drop_first and len(times) > 1:
            times = times[1:]
            encoded_sizes = encoded_sizes[1:]

        raw_bytes = points.nbytes + colors.nbytes
        enc_mean = float(np.mean(encoded_sizes))
        t_mean = float(np.mean(times))
        t_std = float(np.std(times)) if len(times) > 1 else 0.0

        ratio = FileBencher._safe_ratio(raw_bytes, enc_mean)
        bpp = FileBencher._safe_ratio(8.0 * enc_mean, points.shape[0])


        return Metrics(
            str(path),
            int(points.shape[0]),
            int(raw_bytes),
            int(round(enc_mean)),
            t_mean, 
            t_std, 
            ratio,
            bpp
        )