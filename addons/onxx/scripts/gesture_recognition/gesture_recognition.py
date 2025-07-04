from mediapipe import (
    Image,
    ImageFormat
)

from mediapipe.tasks.python.vision import (
    GestureRecognizerOptions,
    GestureRecognizer,
    GestureRecognizerResult,
    RunningMode,
)

from mediapipe.tasks.python import (
    BaseOptions,
)

class GestureRecognition: 

    def __init__(self, model_asset_path: str):

        base_options = BaseOptions(model_asset_path=model_asset_path)

        options = GestureRecognizerOptions(
            base_options=base_options,
            running_mode=RunningMode.LIVE_STREAM,
            result_callback=self.on_result,
        )

        self.recognizer = GestureRecognizer.create_from_options(options)

    
    def on_result(
        self,
        result: GestureRecognizerResult, # type: ignore
        image: Image,
        timestamp_ms: int
    ) -> None:
        # result.gestures is a List[List[Category]]
        for hand_idx, hand_gesture_list in enumerate(result.gestures):
            # pick the top category for this hand
            top_cat = hand_gesture_list[0] if hand_gesture_list else None
            if top_cat:
                print(
                    f"Hand #{hand_idx} @ {timestamp_ms}ms: "
                    f"{top_cat.category_name} ({top_cat.score:.2f})"
                )
    
    def recognize(self, image: Image, frame_timestamp_ms: int):
        self.recognizer.recognize_async(image=image, timestamp_ms=frame_timestamp_ms)

    #Convert the frame received from OpenCV to a MediaPipe Image object.
    def convert_frame(rgb_frame) -> Image:
        return Image(
            image_format=ImageFormat.SRGB,
            data=rgb_frame
        )
