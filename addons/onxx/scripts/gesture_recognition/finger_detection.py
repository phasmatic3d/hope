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

    STRAIGHT_ANGLE_THRESH = 15.0

    # if either joint angle > this, we consider the finger “bent”
    BENT_ANGLE_THRESH = 60.0

    def __init__(
        self, 
        model_asset_path: str,
        num_hands: int,
        running_mode: RunningMode, # type: ignore
        box_size: float=0.05,
        delay_frames: int=15,
    ):

        base_options = BaseOptions(model_asset_path=model_asset_path)

        self.options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            result_callback=self.on_landmarks,
            num_hands=num_hands,
        )

        self.landmarker = HandLandmarker.create_from_options(self.options)
        self.box_size = box_size
        self.latest_bboxes: list[NormalizedBBox | None] = [None] * num_hands
        self._seen_frames = [0] * num_hands
        # approx. half a second at 30 FPS
        self._delay_frames = delay_frames

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

        if not result.hand_landmarks:
            # no hand → reset
            for i in range(self.options.num_hands):
                self._seen_frames[i] = 0
                self.latest_bboxes[i] = None
            return

        detected = result.hand_landmarks
        for i in range(self.options.num_hands):
            if i >= len(detected):
                self._seen_frames[i] = 0
                self.latest_bboxes[i] = None

        # get direction vector
        def to_vec(a, b):
            return (b.x - a.x, b.y - a.y)

        # get the cosine using dot produce formula
        def angle_degree(u, v, eps=1e-6):
            dot_product = u[0]*v[0] + u[1]*v[1]
            normal_u = math.hypot(*u)
            normal_v = math.hypot(*v)
            # if either vector is (almost) zero-length, bait out
            if normal_u < eps or normal_v < eps:
                return 180.0
            # clamp dot/(nu*nv)
            cosθ = max(-1.0, min(1.0, dot_product/(normal_u * normal_v)))
            return math.degrees(math.acos(cosθ))

        def is_finger_straight(angle_1, angle_2):
            return angle_1 < self.STRAIGHT_ANGLE_THRESH and angle_2 < self.STRAIGHT_ANGLE_THRESH

        def is_finger_bent(mcp, pip, dip, tip):
            angle_1 = angle_degree(to_vec(mcp, pip), to_vec(pip, dip))
            angle_2 = angle_degree(to_vec(pip, dip), to_vec(dip, tip))
            # if *either* joint is bent past BENT_ANGLE_THRESH, we call the finger “bent”
            return (angle_1 > self.BENT_ANGLE_THRESH or angle_2 > self.BENT_ANGLE_THRESH)

        # run the for loop in case we want multiple hands later
        for hand_index, hand_landmarks in enumerate(detected):
            # Landmark indices per MediaPipe Hands:
            mcp = hand_landmarks[
                HandLandmark.INDEX_FINGER_MCP
            ]

            pip = hand_landmarks[
                HandLandmark.INDEX_FINGER_PIP
            ]

            dip = hand_landmarks[
                HandLandmark.INDEX_FINGER_DIP
            ]

            tip = hand_landmarks[
                HandLandmark.INDEX_FINGER_TIP
            ]

            vector_1 = to_vec(mcp, pip)
            vector_2 = to_vec(pip, dip)
            vector_3 = to_vec(dip, tip)

            angle_1 = angle_degree(vector_1, vector_2)
            angle_2 = angle_degree(vector_2, vector_3)

            if DEBUG:
                print(f"Segment angles: {angle_1:.1f}°, {angle_2:.1f}°")

            # only count a frame if index finger is straight
            is_index_straight: bool = is_finger_straight(angle_1=angle_1, angle_2=angle_2)

            if not is_index_straight:
                self._seen_frames[hand_index] = 0
                self.latest_bboxes[hand_index] = None
                continue

            middle_finger_bent: bool = is_finger_bent(
                hand_landmarks[HandLandmark.MIDDLE_FINGER_MCP], 
                hand_landmarks[HandLandmark.MIDDLE_FINGER_PIP], 
                hand_landmarks[HandLandmark.MIDDLE_FINGER_DIP],
                hand_landmarks[HandLandmark.MIDDLE_FINGER_TIP],
            )

            if not middle_finger_bent: 
                self._seen_frames[hand_index] = 0
                self.latest_bboxes[hand_index] = None
                continue

            ring_finger_bent: bool = is_finger_bent(                
                hand_landmarks[HandLandmark.RING_FINGER_MCP], 
                hand_landmarks[HandLandmark.RING_FINGER_PIP], 
                hand_landmarks[HandLandmark.RING_FINGER_DIP],
                hand_landmarks[HandLandmark.RING_FINGER_TIP],
            )

            if not ring_finger_bent: 
                self._seen_frames[hand_index] = 0
                self.latest_bboxes[hand_index] = None
                continue

            pinky_finger_bent: bool = is_finger_bent(                
                hand_landmarks[HandLandmark.PINKY_MCP], 
                hand_landmarks[HandLandmark.PINKY_PIP], 
                hand_landmarks[HandLandmark.PINKY_DIP],
                hand_landmarks[HandLandmark.PINKY_TIP],
            ) 

            if not pinky_finger_bent: 
                self._seen_frames[hand_index] = 0
                self.latest_bboxes[hand_index] = None
                continue

            self._seen_frames[hand_index] += 1

            bbox_norm = self._get_normalized_bbox(tip, box_size=self.box_size)

            if DEBUG:
                print(f"  normalized box: {bbox_norm}")

            if DEBUG:
                print(f"Seen Frames: {self._seen_frames[hand_index]} for Hand: {hand_index}")

            if self._seen_frames[hand_index] >= self._delay_frames:
                self.latest_bboxes[hand_index] = bbox_norm
            else:
                self.latest_bboxes[hand_index] = None


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