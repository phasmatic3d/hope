import time
import numpy as np
import encoder as enc # this is the cpp binary
import statslogger as log
from enum import Enum, auto

# Modes for processing
class EncodingMode(Enum):
    FULL = auto()          
    IMPORTANCE = auto()    

class VizualizationMode(Enum):
    COLOR = auto()
    DEPTH = auto()

class DracoWrapper:
    """
    Encoder for 3D point clouds using Draco compression.
    
    Attributes:
        position_quantization_bits (int): Position quantization bits.
        color_quantization_bits (int): Color quantization bits.
        speed_encode (int): Encoder speed [0–10].
        speed_decode (int): Decoder speed [0–10].
        roi_width (int): Width of region-of-interest (unused here).
        roi_height (int): Height of region-of-interest (unused here).
    """
    def __init__(
        self,
        compression_stats: log.CompressionStats = None,
        position_quantization_bits: int = 11,
        color_quantization_bits: int = 8,
        speed_encode: int = 10,
        speed_decode: int = 10,
        roi_width: int = 240,
        roi_height: int = 240,
    ):
        self.position_quantization_bits = position_quantization_bits
        self.color_quantization_bits = color_quantization_bits
        self.speed_encode = speed_encode
        self.speed_decode = speed_decode
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.compression_stats = compression_stats

    def __str__(self) -> str:
        return (
            f"Qpos:{self.pos_quant:2d} "
            f"Qcol:{self.color_quant:2d} "
            f"Spd:{self.speed_encode}/10 "
            f"ROI:{self.roi_width}x{self.roi_height}"
        )

    def encode(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> bytes:

        # This records the number of points in the point-cloud.
        # N×3 array of XYZ coordinates.
        if self.compression_stats != None:
            self.compression_stats.number_of_points = points.shape[0]

        # 3 coordinates, stored as 32-bit floats → 3×4 bytes
        # 3 color channels, stored as 8-bit ints → 3×1 bytes
        if self.compression_stats != None:
            self.compression_stats.raw_bytes = points.nbytes + colors.nbytes

        start = time.perf_counter()
        # note this calls the cpp binary
        buffer = enc.encode_pointcloud(
            points,
            colors,
            self.position_quantization_bits,
            self.color_quantization_bits,
            self.speed_encode,
            self.speed_decode
        )
        end = time.perf_counter()

        if self.compression_stats != None:
            self.compression_stats.compression_ms = (end - start) * 1000
            self.compression_stats.encoded_bytes = len(buffer)


        return buffer