import math
import enum
import numpy as np

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

DEBUG = True

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
class NormalizedBoundingBox:
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

    def to_pixel(self, img_w: int, img_h: int) -> "PixelBoundingBox":
        return PixelBoundingBox(
            x1=int(self.xmin * img_w),
            y1=int(self.ymin * img_h),
            x2=int(self.xmax * img_w),
            y2=int(self.ymax * img_h),
        )

@dataclass
class PixelBoundingBox:
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

class PointingGestureRecognizer:

    STRAIGHT_ANGLE_THRESH = 15.0

    # if either joint angle > this, we consider the finger “bent”
    BENT_ANGLE_THRESH = 60.0

    DISTANCE_WRIST_TO_TIP = 0.20 # meters

    def __init__(
        self, 
        model_asset_path: str,
        num_hands: int,
        running_mode: RunningMode, # type: ignore
        image_width: int,
        image_height: int,
        focal_length_x: float,
        focal_length_y: float,
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
        self.latest_bounding_boxes: list[NormalizedBoundingBox | None] = [None] * num_hands
        self._seen_frames = [0] * num_hands
        # approx. half a second at 30 FPS
        self._delay_frames = delay_frames
        self.image_width = image_width
        self.image_height = image_height
        self.focal_distance_x = focal_length_x
        self.focal_distance_y = focal_length_y
        self.box_size = box_size

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
                self.latest_bounding_boxes[i] = None
            return

        detected = result.hand_landmarks
        for i in range(self.options.num_hands):
            if i >= len(detected):
                self._seen_frames[i] = 0
                self.latest_bounding_boxes[i] = None

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

            wrist = hand_landmarks[
                HandLandmark.WRIST
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
                self.latest_bounding_boxes[hand_index] = None
                continue

            middle_finger_bent: bool = is_finger_bent(
                hand_landmarks[HandLandmark.MIDDLE_FINGER_MCP], 
                hand_landmarks[HandLandmark.MIDDLE_FINGER_PIP], 
                hand_landmarks[HandLandmark.MIDDLE_FINGER_DIP],
                hand_landmarks[HandLandmark.MIDDLE_FINGER_TIP],
            )

            if not middle_finger_bent: 
                self._seen_frames[hand_index] = 0
                self.latest_bounding_boxes[hand_index] = None
                continue

            ring_finger_bent: bool = is_finger_bent(                
                hand_landmarks[HandLandmark.RING_FINGER_MCP], 
                hand_landmarks[HandLandmark.RING_FINGER_PIP], 
                hand_landmarks[HandLandmark.RING_FINGER_DIP],
                hand_landmarks[HandLandmark.RING_FINGER_TIP],
            )

            if not ring_finger_bent: 
                self._seen_frames[hand_index] = 0
                self.latest_bounding_boxes[hand_index] = None
                continue

            pinky_finger_bent: bool = is_finger_bent(                
                hand_landmarks[HandLandmark.PINKY_MCP], 
                hand_landmarks[HandLandmark.PINKY_PIP], 
                hand_landmarks[HandLandmark.PINKY_DIP],
                hand_landmarks[HandLandmark.PINKY_TIP],
            ) 

            if not pinky_finger_bent: 
                self._seen_frames[hand_index] = 0
                self.latest_bounding_boxes[hand_index] = None
                continue

            self._seen_frames[hand_index] += 1

            distance_from_camera = PointingGestureRecognizer.estimate_hand_distance(
                tip=tip,
                wrist=wrist,
                image_width=self.image_width,
                image_height=self.image_height,
                focal_length_x=self.focal_distance_x,
                focal_length_y=self.focal_distance_y
            )

            if DEBUG:
                print(f"Hand #{hand_index} is ~{distance_from_camera:.2f} m from camera")

            #bounding_box_norm = self._get_normalized_bounding_box(tip, box_size=self.box_size)

            bounding_box_norm = self._get_dynamic_bounding_box(
                tip,
                distance_from_camera,
            )

            if DEBUG:
                print(f"  normalized box: {bounding_box_norm}")

            if DEBUG:
                print(f"Seen Frames: {self._seen_frames[hand_index]} for Hand: {hand_index}")

            if self._seen_frames[hand_index] >= self._delay_frames:
                self.latest_bounding_boxes[hand_index] = bounding_box_norm
            else:
                self.latest_bounding_boxes[hand_index] = None


    def _get_dynamic_bounding_box(
        self,
        landmark,
        distance_z: float,
    ) -> NormalizedBoundingBox:
        # if depth invalid/far, fall back the fixed normalized size:
        if distance_z <= 0 or not math.isfinite(distance_z):
            half_norm = self.box_size / 2
            return NormalizedBoundingBox(
                landmark.x - half_norm,
                landmark.y - half_norm,
                landmark.x + half_norm,
                landmark.y + half_norm,
            )

        # compute half-width in *pixels* via pinhole:  half_px = (f * W_real/2) / Z
        half_px_x = (self.focal_distance_x * self.box_size/2) / distance_z
        half_px_y = (self.focal_distance_y * self.box_size/2) / distance_z

        # normalize by image dims to get [0..1] units
        half_norm_x = half_px_x / self.image_width
        half_norm_y = half_px_y / self.image_height

        # clamp to [0..1]
        xmin = max(landmark.x - half_norm_x, 0.0)
        ymin = max(landmark.y - half_norm_y, 0.0)
        xmax = min(landmark.x + half_norm_x, 1.0)
        ymax = min(landmark.y + half_norm_y, 1.0)

        return NormalizedBoundingBox(xmin, ymin, xmax, ymax)

    def get_masks(self, frame_shape: tuple[int,int]) -> "np.ndarray":
        """
        Returns a stack of binary masks (one per hand) where the pixel
        inside each debounced fingertip BBox = 1, elsewhere = 0.

        Args:
          frame_shape: (height, width) of the frame you used for inference.

        Returns:
          masks: np.ndarray of shape (num_hands, height, width), dtype=uint8.
        """

        height, width = frame_shape[:2]
        masks = np.zeros((self.options.num_hands, height, width), dtype=np.uint8)

        for i, bounding_box_norm in enumerate(self.latest_bounding_boxes):
            if bounding_box_norm is None:
                continue
            bounding_box_pixel_space = bounding_box_norm.to_pixel(img_w=width, img_h=height)
            masks[i, bounding_box_pixel_space.y1:bounding_box_pixel_space.y2, bounding_box_pixel_space.x1:bounding_box_pixel_space.x2] = 1

        return masks

    def _get_normalized_bounding_box(self, landmark, box_size: float) -> NormalizedBoundingBox:
        """
        Returns (xmin, ymin, xmax, ymax) in normalized coords [0..1].
        box_size is fraction of the image (e.g. 0.05 = 5% of width/height).
        """
        half = box_size / 2
        xmin = max(landmark.x - half, 0.0)
        ymin = max(landmark.y - half, 0.0)
        xmax = min(landmark.x + half, 1.0)
        ymax = min(landmark.y + half, 1.0)
        return NormalizedBoundingBox(xmin, ymin, xmax, ymax)

    #Convert the frame received from OpenCV to a MediaPipe Image object.
    @staticmethod
    def convert_frame(rgb_frame) -> Image:
        return Image(
            image_format=ImageFormat.SRGB,
            data=rgb_frame
        )

    @staticmethod
    def estimate_hand_distance(
        tip: float, 
        wrist: float,
        image_width: int, 
        image_height: int,
        focal_length_x: float, 
        focal_length_y: float,
    ) -> float:

        x1, y1 = int(tip.x * image_width), int(tip.y * image_height)
        x2, y2 = int(wrist.x * image_width), int(wrist.y * image_height)

        # Compute pixel‐space distance
        distance_in_pixels = math.hypot(x2 - x1, y2 - y1)
        if distance_in_pixels <= 0:
            return float("inf")  # or some fallback

        # Use average focal length
        focal_length_pixels = (focal_length_x + focal_length_y) / 2.0

        # Pinhole model: Z = f * D_real / d_px
        z = (focal_length_pixels * PointingGestureRecognizer.DISTANCE_WRIST_TO_TIP) / distance_in_pixels
        return z