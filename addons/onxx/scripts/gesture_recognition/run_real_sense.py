import time

import cv2
import numpy as np
import pyrealsense2 as rs

from realsense import (
    RSConfigBuilder
)

from gesture_recognition import GestureRecognition


# Configure depth and color streams
# Get device product line for setting a supporting resolution
#pipeline_wrapper = rs.pipeline_wrapper(pipeline)
#pipeline_profile = config.resolve(pipeline_wrapper)
#device = pipeline_profile.get_device()
#device_product_line = str(device.get_info(rs.camera_info.product_line))


pipeline = rs.pipeline()

RSConfigBuilder.get_devices()
builder = RSConfigBuilder(serial="243322072106")  
builder \
.enable_stream(
   stream_type=rs.stream.depth,
   index=0,
   resolution=(640, 480),
   fmt=rs.format.z16,
   fps=30
) \
.enable_stream(
   stream_type=rs.stream.color,
   index=0,
   resolution=(640, 480),
   fmt=rs.format.bgr8,
   fps=30
) \
.enable_stream(
   stream_type=rs.stream.infrared,
   index=1,
   resolution=(640, 480),
   fmt=rs.format.y8,
   fps=30
) \
.enable_stream(
   stream_type=rs.stream.infrared,
   index=2,
   resolution=(640, 480),
   fmt=rs.format.y8,
   fps=30
) \
.enable_record_to_file("test.bag")

cfg = builder.resolve_with_pipeline(pipeline=pipeline)

print(builder.get_device())
print(builder.get_all_stream_profiles())
print(builder.get_stream_profile(rs.stream.depth, 0))

pipeline.start(cfg)

recognizer = GestureRecognition(model_asset_path="gesture_recognizer.task")
