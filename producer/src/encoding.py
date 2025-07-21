import threading
from enum import Enum, auto
import time
import numpy as np

import encoder
from typing import Tuple


# Modes for processing
class Mode(Enum):
    FULL = auto()          
    IMPORTANCE = auto()    

class VizMode(Enum):
    COLOR = auto()
    DEPTH = auto()


class Timer:
    def __init__(self):
        self.reset()
    def reset(self):
        self.start = time.time()
    def elapsed_ms(self):
        return (time.time() - self.start) * 1000


class EncodingStats:
    def __init__(self):
        self.encode_ms = 0
        self.pts = 0
        self.raw_bytes = 0
        self.encoded_bytes = 0

class GeneralStats:
    def __init__(self):
        self.frame_ms = 0
        self.cull_ms = 0
        self.pc_ms = 0
        self.prep_ms = 0
        self.true_enc_ms = 0
        self.det_ms = 0
        self.gesture_recognition_ms = 0

    def get_total_time(self):
        return self.pc_ms + self.prep_ms + self.det_ms + self.gesture_recognition_ms


class DracoEncoder:
    def __init__(self):
        self.posQuant = 10
        self.colorQuant = 8
        self.speedEncode = 9
        self.speedDecode = 1
        self.roiWidth = 240
        self.roiHeight = 240
    def to_string(self):
        return f"Qpos:{self.posQuant:2d} Qcol:{self.colorQuant:2d} Spd:{self.speedEncode}/10 ROI:{self.roiWidth}x{self.roiHeight}"

    def encode(self, points, colors, stats: EncodingStats):

        stats.pts = points.shape[0]
        stats.raw_bytes = stats.pts * (3 * 4 + 3 * 1)

        buffer = encoder.encode_pointcloud(
            points,
            colors,
            self.posQuant,
            self.colorQuant,
            self.speedEncode,
            self.speedDecode
        )

        return buffer

def _encode_chunk(pts: np.ndarray,
                colors: np.ndarray,
                encoder: DracoEncoder,
                stats: EncodingStats
                ) -> Tuple[bytes, EncodingStats]:
# instantiate a fresh encoder in this process
    start = time.time()
    buf = encoder.encode(pts, colors, stats)
    end = time.time()
    stats.encode_ms = (end - start) * 1000
    return buf, stats
