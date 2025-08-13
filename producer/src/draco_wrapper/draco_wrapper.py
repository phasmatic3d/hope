import time
import numpy as np
from . import draco_bindings as dcb

from enum import Enum, auto

# Modes for processing
class EncodingMode(Enum):
    NONE = auto()
    FULL = auto()          
    IMPORTANCE = auto()  

class VisualizationMode(Enum):
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
        deduplicate: bool
    ) -> bytes:

        if points.size == 0: 
            return b"";
    
        # note this calls the cpp binary
        buffer = dcb.encode_pointcloud(
            points,
            colors,
            self.position_quantization_bits,
            self.color_quantization_bits,
            self.speed_encode,
            self.speed_decode,
            deduplicate
        )
        return buffer