from collections.abc import Iterable
from rich.table import Table
import os

import pandas as pd
from datetime import datetime
from collections import deque

from typing import Tuple

from draco_wrapper import (
    EncodingMode
)
from stats import (
    CompressionStats,
    PipelineTiming
)


CSV_DIR     = "stats"
FULL_CSV    = os.path.join(CSV_DIR, "stats_full.csv")
IMP_CSV     = os.path.join(CSV_DIR, "stats_importance.csv")
os.makedirs(CSV_DIR, exist_ok=True)


def _append_row(path: str, row: dict):
    write_header = not os.path.exists(path)
    df = pd.DataFrame([row]).round(2)
    df.to_csv(path, mode="a", header=write_header, index=False)
    print(f"Appended stats to {path}")
# CSV WRITER FUNCTION
def write_stats_csv(
    stats_buffer: deque,
    mode,
    encoding_speed: int,
    pos_quant_bits: int,
    clr_res: Tuple[int,int],
    depth_res: Tuple[int,int],
    active_layers: list[bool]
):
    """
    Build one summary row from the last N frames and append it
    to either the IMPORTANCE CSV or the FULL CSV.
    """
    if len(stats_buffer) < stats_buffer.maxlen:
        print("No stats to write yet.")
        return

    df  = pd.DataFrame(stats_buffer)
    avg = df.mean()

    # common fields
    common = {
        "timestamp":              datetime.now().isoformat(),
        "mode":                   mode.name if hasattr(mode, "name") else str(mode),
        "encoding_speed":         encoding_speed,
        "position_quant_bits":    pos_quant_bits,
        "color_resolution":       f"{clr_res[0]}x{clr_res[1]}",
        "depth_resolution":       f"{depth_res[0]}x{depth_res[1]}",
        #"avg_frame_alignment_ms": avg.get("frame_alignment_ms",    pd.NA),
        #"avg_depth_culling_ms":   avg.get("depth_culling_ms",      pd.NA),
        "frame_preparation_ms":         avg.get("frame_preparation_ms",      pd.NA),
        "data_preparation_ms"           :avg.get("data_preparation_ms",   pd.NA),  
        "one_way_ms"                : avg.get("one_way_ms",            pd.NA),
        "one_way_plus_processing_ms"   : avg.get("one_way_plus_processing_ms",    pd.NA),
        "layer0_on":               active_layers[0],
        "layer1_on":               active_layers[1],
        "layer2_on":               active_layers[2],
    }

    if mode == EncodingMode.FULL:
        row  = {**common,
                "encode_ms": avg.get("full_encode_ms", pd.NA),
                "points": int(avg.get("full_points", pd.NA)),}
        path = FULL_CSV
    else:  # IMPORTANCE
        row  = {**common,
                "roi_encode_ms":                avg.get("roi_encode_ms",               pd.NA),
                "outside_encode_ms":            avg.get("outside_encode_ms",           pd.NA),
                "multiprocessing_compression_ms": avg.get("multiprocessing_compression_ms", pd.NA),
                "in_roi_points": int(avg.get("in_roi_points", pd.NA)),
                "out_roi_points": int(avg.get("out_roi_points", pd.NA)),
               }
        path = IMP_CSV
    

    _append_row(path, row)

    stats_buffer.clear()


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