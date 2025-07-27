import math 

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