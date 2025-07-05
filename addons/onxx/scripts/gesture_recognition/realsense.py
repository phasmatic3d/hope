import pyrealsense2 as rs

from typing import Optional, Tuple

class RSConfigBuilder:
    def __init__(self, serial: Optional[str] = None):
        """
        Create a fresh config. If `serial` is given, we'll bind to that device.
        """
        self._cfg = rs.config()
        self._pipeline_profile = None
        if serial:
            # select a specific RealSense device by serial
            self._cfg.enable_device(serial)
    
    def enable_record_to_file(
        self,
        file_name: str
    ) -> "RSConfigBuilder":
        """
        Record all enabled streams into a .bag file as you stream.
        """
        self._cfg.enable_record_to_file(file_name)
        return self

    def enable_stream(
        self,
        stream_type: rs.stream,
        index: int,
        resolution: Tuple[int, int] = (640, 480),
        fmt: rs.format = rs.format.z16,
        fps: int = 30,
    ) -> "RSConfigBuilder":
        """
        Enable one stream. You can call this multiple times to add more.
        """
        width, height = resolution
        self._cfg.enable_stream(
            stream_type, 
            index,
            width,
            height, 
            fmt, 
            fps,
        )
        return self

    def disable_stream(
        self,
        stream_type: rs.stream,
        index: int
    ) -> "RSConfigBuilder":
        """
        Disable a specific stream if previously enabled (or to clear defaults).
        """
        self._cfg.disable_stream(stream_type, index)
        return self

    def enable_all_streams(self) -> "RSConfigBuilder":
        self._cfg.enable_all_streams()
        return self

    def disable_all_streams(self) -> "RSConfigBuilder":
        self._cfg.disable_all_streams()
        return self

    def resolve_with_pipeline(self, pipeline: rs.pipeline) -> rs.config:
        """
        Internally resolves filters and checks compatibility with the given pipeline.
        Returns the ready-to-use rs.config object.
        """
        wrapper = rs.pipeline_wrapper(pipeline)
        self._pipeline_profile = self._cfg.resolve(wrapper)
        return self._cfg

    def get_pipeline_profile(self) -> rs.pipeline_profile:
        if self._pipeline_profile is None:
            raise RuntimeError("You must call resolve_with_pipeline() before asking for the profile.")
        return self._pipeline_profile

    def get_device(self) -> rs.device:
        """Shortcut to the active device object."""
        return self.get_pipeline_profile().get_device()

    def get_stream_profile(
        self,
        stream_type: rs.stream,
        stream_index: int = 0
    ) -> rs.stream_profile:
        """Return the rs.stream_profile for one enabled stream."""
        return self.get_pipeline_profile().get_stream(stream_type, stream_index)

    def get_all_stream_profiles(self) -> list[rs.stream_profile]:
        """Return a list of all enabled stream profiles."""
        return list(self.get_pipeline_profile().get_streams())
    
    def build(self, pipeline: rs.pipeline) -> rs.config:
        """
        Shortcut for resolve_with_pipeline().
        """
        return self.resolve_with_pipeline(pipeline)

    @staticmethod
    def get_devices():
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            name   = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            print(f"Detected Device: {name} — Serial: {serial}")

class RealsenseStream: 

    def __init__(self):
        pass

    def init_config(
        file_name:str,
        repeat_playback: bool,
    ):

        rs.config()
