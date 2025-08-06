import math 
import struct
import os
import numpy as np

# Function to calculate focal length in pixels from the field of view and resolution width
def calculate_focal_length_in_pixels(
    resolution_width: float, 
    resolution_height: float,
    focal_length_x: float,
    focal_length_y: float,
) -> tuple[float, float]:
    
    #Compute horizontal & vertical FOV (in degrees)
    #FOV = 2 * arctan( size_sensor / (2 * focal_length) )
    #but in pixel-space that becomes:
    hfov = 2 * math.degrees(math.atan(resolution_width / (2 * focal_length_x)))
    vfov = 2 * math.degrees(math.atan(resolution_height / (2 * focal_length_y)))

    return hfov, vfov


def estimate_distance():
    pass

def write_pointcloud_ply(filename: str,
                         vertices: np.ndarray,   # shape (N,3), dtype float32
                         colors:   np.ndarray     # shape (N,3), dtype uint8
                        ) -> None:
    """
    Write a binary‐little‐endian PLY file with position+color per vertex.
    """
    N = vertices.shape[0]
    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
        ""
    ]).encode("utf-8")

    # Ensure the output dir exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as f:
        f.write(header)
        # pack each vertex as 3×float + 3×uchar
        for (x,y,z), (r,g,b) in zip(vertices, colors):
            f.write(struct.pack("<fffBBB", x, y, z, r, g, b))