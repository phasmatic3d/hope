from collections.abc import Iterable
from rich.table import Table

class CompressionStats:
    def __init__(self):
        self.compression_ms   = 0
        self.number_of_points = 0
        self.raw_bytes        = 0
        self.encoded_bytes    = 0
    
    def make_table(self, section: str, show_headers: bool = True) -> Table:
        title = f"==== {section} ===="
        table = Table(title=title, box=None, padding=(0,1), show_header=show_headers)

        for name, value in vars(self).items():
            # Turn "encode_ms" → "Encode Ms", "raw_bytes" → "Raw Bytes", etc.
            label = name.replace('_', ' ').title()

            # Format floats as milliseconds, leave ints alone
            if isinstance(value, float):
                display = f"{value:.2f} ms"
            else:
                display = str(value)

            table.add_row(label, display)

        return table

    def __str__(self):
        lines = [
            f"{name}: {value}"
            for name, value in vars(self).items()
        ]
        return "\n".join(lines)

    def get_total_time(self):
        # sum every attribute whose name ends with "_ms"
        return sum(
            value
            for name, value in vars(self).items()
            if name.endswith('_ms')
        )

class PipelineTiming:
    def __init__(self):
        self.frame_alignment_ms             = 0
        self.point_cloud_creation_ms        = 0
        self.depth_culling_ms               = 0
        self.gesture_recognition_ms         = 0
        self.data_preparation_ms            = 0
        self.texture_scaling_ms             = 0
        self.build_valid_points_ms          = 0
        self.build_mask_for_roi_ms          = 0
        self.multiprocessing_compression_ms = 0
        self.sam2_ms = 0

    def get_total_time(self):
        # sum every attribute whose name ends with "_ms"
        return sum(
            value
            for name, value in vars(self).items()
            if name.endswith('_ms') and name != "texture_scaling_ms" and name != "build_valid_points_ms" and name != "build_mask_for_roi_ms"
            and name != "multiprocessing_compression_ms"
        )

    def __str__(self):
        # nicely print all *_ms fields without hard-coding
        lines = [
            f"{name:30s}: {value:8.2f} ms"
            for name, value in vars(self).items()
            if name.endswith('_ms')
        ]
        return "\n".join(lines)

    def make_table(self, section: str, show_headers: bool = True) -> Table:
        title = f"==== {section} ===="

        table = Table(
            title=title, 
            box=None, 
            padding=(0,1), 
            show_header=show_headers
        )

        # Dynamically pull in all "<something>_ms" attrs
        for attr, value in vars(self).items():
            if not attr.endswith('_ms'):
                continue
            # turn "point_cloud_creation_ms" → "Point Cloud Creation"
            label = attr[:-3].replace('_', ' ').title()
            table.add_row(label, f"{value:.2f} ms")

        return table

def calculate_overall_time(
    pipeline_stats: PipelineTiming,
    compression_stats: Iterable[CompressionStats] | CompressionStats,   # accept one or many
) -> float:
    """
    Return the sum of all *_ms timings from the pipeline plus every
    CompressionStats object supplied.
    """

    # ── Normalise to an iterable ────────────────────────────────────
    if isinstance(compression_stats, CompressionStats):
        # caller passed a single object; wrap it so we can iterate
        compression_stats = [compression_stats]

    # If it still isn’t iterable, raise immediately so the bug is obvious
    if not isinstance(compression_stats, Iterable):
        raise TypeError(
            "compression_stats must be a CompressionStats instance "
            "or an iterable of CompressionStats"
        )

    # ── Aggregate timings ───────────────────────────────────────────
    pipeline_time = pipeline_stats.get_total_time()
    compression_time = sum(cs.get_total_time() for cs in compression_stats)

    return pipeline_time + compression_time

def make_total_time_table(total_time: float, section: str = "Overall Total") -> Table:
    """
    Wraps a single total‐time value into a Rich Table.
    """
    title = f"==== {section} ===="
    table = Table(title=title, box=None, padding=(0,1), show_header=False)
    table.add_row("Total Time", f"{total_time:.2f} ms")
    return table