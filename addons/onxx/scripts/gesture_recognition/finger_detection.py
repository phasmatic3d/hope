import math
import enum
import cv2

from dataclasses import dataclass

from mediapipe import (
    Image,
    ImageFormat
)

from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode,
)

from mediapipe.tasks.python import BaseOptions

DEBUG = False

#copied from the library directly
class HandLandmark(enum.IntEnum):
    """The 21 hand landmarks."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

@dataclass
class NormalizedBBox:
    """A bounding box in normalized coordinates [0.0–1.0]."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    def to_pixel(self, img_w: int, img_h: int) -> "PixelBBox":
        return PixelBBox(
            x1=int(self.xmin * img_w),
            y1=int(self.ymin * img_h),
            x2=int(self.xmax * img_w),
            y2=int(self.ymax * img_h),
        )

@dataclass
class PixelBBox:
    """A bounding box in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

class FingerDirection:

    def __init__(
        self, 
        model_asset_path: str,
        num_hands: int,
        running_mode: RunningMode, # type: ignore
        box_size: float=0.05
    ):

        base_options = BaseOptions(model_asset_path=model_asset_path)

        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            result_callback=self.on_landmarks,
            num_hands=num_hands,
        )

        self.landmarker = HandLandmarker.create_from_options(options)
        self.box_size = box_size
        self.latest_bbox: NormalizedBBox | None = None
        # approx. half a second at 30 FPS
        self._delay_frames = 15
        self._seen_frames = 0

    def recognize(self, image: Image, frame_timestamp_ms: int):
        self.landmarker.detect_async(image=image, timestamp_ms=frame_timestamp_ms)

    def on_landmarks(
        self,
        result: HandLandmarkerResult,  # type: ignore
        image: Image,
        timestamp_ms: int
    ) -> None:
        """
        Callback invoked with each frame’s hand landmarks. Computes and prints
        the pointing direction of the index finger for each detected hand.
        """

        for _, hand_landmarks in enumerate(result.hand_landmarks):
            # Landmark indices per MediaPipe Hands:
            # INDEX_FINGER_TIP = 8

            tip = hand_landmarks[
                HandLandmark.INDEX_FINGER_TIP
            ]

            bbox_norm = self._get_normalized_bbox(tip, box_size=self.box_size)

            if DEBUG:
                print(f"  normalized box: {bbox_norm}")

            #self._seen_frames += 1
            #if self._seen_frames >= self._delay_frames:
            #    self.latest_bbox = bbox_norm
            #else:
            #    self._seen_frames = 0
            #    self.latest_bbox = None


    def _get_normalized_bbox(self, landmark, box_size: float) -> NormalizedBBox:
        """
        Returns (xmin, ymin, xmax, ymax) in normalized coords [0..1].
        box_size is fraction of the image (e.g. 0.05 = 5% of width/height).
        """
        half = box_size / 2
        xmin = max(landmark.x - half, 0.0)
        ymin = max(landmark.y - half, 0.0)
        xmax = min(landmark.x + half, 1.0)
        ymax = min(landmark.y + half, 1.0)
        return NormalizedBBox(xmin, ymin, xmax, ymax)

    #Convert the frame received from OpenCV to a MediaPipe Image object.
    @staticmethod
    def convert_frame(rgb_frame) -> Image:
        return Image(
            image_format=ImageFormat.SRGB,
            data=rgb_frame
        )