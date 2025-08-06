from collections.abc import Iterable
from rich.table import Table
import os

import pandas as pd
from datetime import datetime
from collections import deque

from typing import List, Dict, Tuple

from draco_wrapper.draco_wrapper import (
    EncodingMode
)
from stats.stats import (
    CompressionStats,
    PipelineTiming
)

from itertools import product



CSV_DIR     = "stats"
PERF_NONE_CSV  = os.path.join(CSV_DIR, "stats_none_performance.csv")

PERF_FULL_CSV  = os.path.join(CSV_DIR, "stats_full_performance.csv")
PERF_IMP_CSV   = os.path.join(CSV_DIR, "stats_importance_performance.csv")
QUAL_FULL_CSV  = os.path.join(CSV_DIR, "stats_full_quality.csv")
QUAL_IMP_CSV   = os.path.join(CSV_DIR, "stats_importance_quality.csv")
os.makedirs(CSV_DIR, exist_ok=True)

# where to write simulation results:
SIM_CSV = os.path.join(CSV_DIR, "stats_simulation.csv")


def _append_row(path: str, row: dict): 
    ERROR_COLS = ["mean_pos_error", "max_pos_error", "mean_color_error"]

    write_header = not os.path.exists(path)
    df = pd.DataFrame([row])

    # 1) Pre‐format your error columns to 6dp strings
    for col in ERROR_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.6f}")

    # 2) Write out:
    #    - use float_format for *all other* floats → 2dp
    df.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
        float_format="%.2f"
    )
    print(f"Appended stats to {path}")
# CSV WRITER FUNCTION
def write_stats_csv(
    stats_buffer: deque,
    mode: EncodingMode,
    clr_res: Tuple[int,int],
    depth_res: Tuple[int,int],
    # for FULL
    encoding_speed: int = None,
    pos_quant_bits: int = None,
    active_layers: list[bool] = None,
    # for IMPORTANCE
    encoding_speed_in: int = None,
    pos_quant_bits_in: int = None,
    encoding_speed_out: int = None,
    pos_quant_bits_out: int = None,
):
    # need 30 frames before writing
    if len(stats_buffer) < stats_buffer.maxlen:
        print(f"Waiting for {stats_buffer.maxlen} frames (have {len(stats_buffer)})")
        return

    avg = pd.DataFrame(stats_buffer).mean()
    stats_buffer.clear()

    # shared fields
    ts = datetime.now().isoformat()
    common = {
        "timestamp":         ts,
        "color_resolution":  f"{clr_res[0]}x{clr_res[1]}",
        "depth_resolution":  f"{depth_res[0]}x{depth_res[1]}",
        "mode":              mode.name,
    }

    if mode == EncodingMode.NONE:
        row = {
            **common,
            "points":                      int(avg.get("num_points", pd.NA)),
            "frame_preparation_ms":        avg.get("frame_preparation_ms",   pd.NA),
            "data_preparation_ms":         avg.get("data_preparation_ms",    pd.NA),
            "one_way_ms":                  avg.get("one_way_ms",             pd.NA),
            "geometry_upload_ms":  avg.get("geometry_upload_ms", pd.NA),
        }
        # total_time = frame_prep + data_prep + one_way+proc
        row["total_time_ms"] = (
            row["frame_preparation_ms"]
          + row["data_preparation_ms"]
          + row["one_way_ms"]
          + row["geometry_upload_ms"]
        )
        path = PERF_NONE_CSV

    elif mode == EncodingMode.FULL:
        row = {
            **common,
            "encoding_speed":        encoding_speed,
            "position_quant_bits":   pos_quant_bits,
            "layer0_on":             active_layers[0],
            "layer1_on":             active_layers[1],
            "layer2_on":             active_layers[2],
            "points":                int(avg.get("full_points", pd.NA)),
            "frame_preparation_ms":  avg.get("frame_preparation_ms",   pd.NA),
            "data_preparation_ms":   avg.get("data_preparation_ms",    pd.NA),
            "encode_ms":             avg.get("full_encode_ms",         pd.NA),
            "one_way_ms":            avg.get("one_way_ms",             pd.NA),
            "decode_ms": avg.get("decode_ms", pd.NA),
        }
        # total_time = frame_prep + data_prep + encode + one_way+proc
        row["total_time_ms"] = (
            row["frame_preparation_ms"]
          + row["data_preparation_ms"]
          + row["one_way_ms"]
          + row["encode_ms"]
          + row["decode_ms"]
        )
        path = PERF_FULL_CSV

    else:  # IMPORTANCE
        in_pts  = int(avg.get("in_roi_points",  pd.NA))
        out_pts = int(avg.get("out_roi_points", pd.NA))
        row = {
            **common,
            "encoding_speed_in":       encoding_speed_in,
            "position_quant_bits_in":  pos_quant_bits_in,
            "encoding_speed_out":      encoding_speed_out,
            "position_quant_bits_out": pos_quant_bits_out,
            "layer0_on":               active_layers[0],
            "layer1_on":               active_layers[1],
            "layer2_on":               active_layers[2],
            "points_in":               in_pts,
            "points_out":              out_pts,
            "points":                  in_pts + out_pts,
            "frame_preparation_ms":    avg.get("frame_preparation_ms",   pd.NA),
            "data_preparation_ms":     avg.get("data_preparation_ms",    pd.NA),
            "encode_ms":               avg.get("multiprocessing_compression_ms", pd.NA),
            "one_way_ms":              avg.get("one_way_ms",             pd.NA),
            "decode_ms":               avg.get("decode_ms", pd.NA),
            "geometry_upload_ms":      avg.get("geometry_upload_ms", pd.NA),
            "render_ms":               avg.get("render_ms", pd.NA),

        }

        row["total_time_ms"] = (
            row["frame_preparation_ms"]
          + row["data_preparation_ms"]
          + row["encode_ms"]
          + row["one_way_ms"]
          + row["decode_ms"]
          + row["geometry_upload_ms"]
          + row["render_ms"]
        )
        path = PERF_IMP_CSV

    

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


# example search‐spaces
FULL_POS_BITS  = [10, 11, 12]
FULL_COL_BITS          = [8]
FULL_ENCODING_SPEED    = [0, 5, 10]
FULL_DECODING_SPEED    = [0, 5, 10]


ROI_POS_BITS   = [10, 11, 12]
ROI_COL_BITS   = [7,8]
ROI_ENCODING_SPEED     = [0, 5, 10]
ROI_DECODING_SPEED     = [0, 5, 10]

OUT_POS_BITS   = [8, 9, 10]
OUT_COL_BITS   = [6,7]
OUT_ENCODING_SPEED     = [5, 10]
OUT_DECODING_SPEED     = [5, 10]
# all 3-bit on/off combos except the all-False case
LAYER_OPTIONS = [
    combo
    for combo in product([True, False], repeat=3)
    if any(combo)    # drop (False, False, False)
]

def generate_combinations(mode):
    if mode == EncodingMode.FULL:
        return [
            {"pos_bits": p, "col_bits": c, "encoding_speed": enc_s, "decoding_speed": dec_s}
            for p, c, enc_s, dec_s in product(FULL_POS_BITS, FULL_COL_BITS, FULL_ENCODING_SPEED, FULL_DECODING_SPEED)
        ]
    elif mode == EncodingMode.IMPORTANCE:
        combos = []
        for quant_pos_roi, quant_col_roi, encoding_speed_roi, decoding_speed_roi, \
            quant_pos_out, quant_col_out, encoding_speed_out, decoding_speed_out, layers in product(
            ROI_POS_BITS, ROI_COL_BITS, ROI_ENCODING_SPEED, ROI_DECODING_SPEED,
            OUT_POS_BITS, OUT_COL_BITS, OUT_ENCODING_SPEED, OUT_DECODING_SPEED, LAYER_OPTIONS
        ):
            combos.append({
                "pos_bits_in":  quant_pos_roi,
                "col_bits_in":  quant_col_roi,
                "encoding_speed_in":     encoding_speed_roi,
                "decoding_speed_in":     decoding_speed_roi,
                "pos_bits_out":          quant_pos_out,
                "col_bits_out":          quant_col_out,
                "encoding_speed_out":    encoding_speed_out,
                "decoding_speed_out":    decoding_speed_out,
                "layers":       layers,
            })
        return combos
    else:
        return []
    
def write_simulation_csv(
    sim_buffer: deque,
    combos: List[Dict],
    combo_index: int,
    mode: EncodingMode,
    clr_res: Tuple[int,int],
    depth_res: Tuple[int,int],
):
    """
    Once sim_buffer has `maxlen` entries, average them, write one row
    for combos[combo_index], and return the next combo_index (or None if done).
    """
    # only fire when full
    if len(sim_buffer) < sim_buffer.maxlen:
        return combo_index

    # average the 90 frames
    avg = pd.DataFrame(sim_buffer).mean()
    sim_buffer.clear()

    #path
    if mode == EncodingMode.NONE:
        path = PERF_NONE_CSV
    elif mode == EncodingMode.FULL:
        path = PERF_FULL_CSV
    else:  # IMPORTANCE
        path = PERF_IMP_CSV

    # shared fields
    timestamp = datetime.now().isoformat()
    common = {
        "experiment_id":    combo_index,
        "timestamp":        timestamp,
        "color_resolution": f"{clr_res[0]}x{clr_res[1]}",
        "depth_resolution": f"{depth_res[0]}x{depth_res[1]}",
        "mode":             mode.name,
    }

    combo = combos[combo_index]
    # start building the row with common + combo hyperparams
    row = { **common }

    # inject combo fields in strict order:
    if mode == EncodingMode.NONE:
        # NONE has no hyperparams beyond common
        pass

    elif mode == EncodingMode.FULL:
        # full‐mode hyperparams
        row["encoding_speed"]      = combo["encoding_speed"]
        row["decoding_speed"]      = combo["decoding_speed"]
        row["position_quant_bits"] = combo["pos_bits"]
        row["color_quant_bits"]    = combo["col_bits"]

    else:  # IMPORTANCE
        # importance hyperparams
        row["position_quant_bits_in"]  = combo["pos_bits_in"]
        row["color_quant_bits_in"]  = combo["col_bits_in"]
        row["encoding_speed_in"]       = combo["encoding_speed_in"]
        row["decoding_speed_in"]       = combo["decoding_speed_in"]

        row["position_quant_bits_out"] = combo["pos_bits_out"]
        row["color_quant_bits_out"] = combo["col_bits_out"]
        row["encoding_speed_out"]      = combo["encoding_speed_out"]
        row["decoding_speed_out"]      = combo["decoding_speed_out"]
        
        # now the layers tuple
        for i, on in enumerate(combo["layers"]):
            row[f"layer{i}_on"] = on

    # now the averaged statistics, in the same order as write_stats_csv:
    if mode == EncodingMode.NONE:
        # points = total number of pts in sim_buffer: use avg.num_points
        row["points"]                       = int(avg.get("num_points", pd.NA))
        row["frame_preparation_ms"]         = avg.get("frame_preparation_ms", pd.NA)
        row["data_preparation_ms"]          = avg.get("data_preparation_ms",    pd.NA)
        row["one_way_ms"]                   = avg.get("one_way_ms",             pd.NA)
        row["geometry_upload_ms"]   = avg.get("geometry_upload_ms", pd.NA)
        row["render_ms"]                   = avg.get("render_ms", pd.NA)

        # total_time
        row["total_time_ms"] = (
            row["frame_preparation_ms"]
          + row["data_preparation_ms"]
          + row["one_way_ms"]
          + row["geometry_upload_ms"]
          + row["render_ms"]  
        )

    elif mode == EncodingMode.FULL:
        row["points"]                       = int(avg.get("full_points", pd.NA))
        row["frame_preparation_ms"]         = avg.get("frame_preparation_ms", pd.NA)
        row["data_preparation_ms"]          = avg.get("data_preparation_ms",    pd.NA)
        row["encode_ms"]                    = avg.get("full_encode_ms",         pd.NA)
        row["one_way_ms"]                   = avg.get("one_way_ms",             pd.NA)
        row["decode_ms"]                    = avg.get("decode_ms", pd.NA)
        row["geometry_upload_ms"]            = avg.get("geometry_upload_ms", pd.NA)
        row["render_ms"]                     = avg.get("render_ms", pd.NA)

        row["total_time_ms"] = (
            row["frame_preparation_ms"]
          + row["data_preparation_ms"]
          + row["encode_ms"]
          + row["one_way_ms"]
          + row["decode_ms"]
          + row["geometry_upload_ms"]
          + row["render_ms"]  
        )

    else:  # IMPORTANCE
        in_pts  = int(avg.get("in_roi_points",  pd.NA))
        out_pts = int(avg.get("out_roi_points", pd.NA))
        row["points_in"]                    = in_pts
        row["points_out"]                   = out_pts
        row["points"]                       = in_pts + out_pts
        row["frame_preparation_ms"]         = avg.get("frame_preparation_ms",   pd.NA)
        row["data_preparation_ms"]          = avg.get("data_preparation_ms",    pd.NA)
        row["roi_encode_ms"]                    = avg.get("roi_encode_ms", pd.NA)
        row["out_encode_ms"]                    = avg.get("out_encode_ms", pd.NA)
        row["encode_ms"]                    = avg.get("multiprocessing_compression_ms", pd.NA)
        row["one_way_ms"]                   = avg.get("one_way_ms",             pd.NA)
        row["decode_ms"]                    = avg.get("decode_ms", pd.NA)
        row["geometry_upload_ms"]           = avg.get("geometry_upload_ms", pd.NA)
        row["render_ms"]                   = avg.get("render_ms", pd.NA)

        row["total_time_ms"] = (
            row["frame_preparation_ms"]
          + row["data_preparation_ms"]
          + row["encode_ms"]
          + row["one_way_ms"]
          + row["decode_ms"]
          + row["geometry_upload_ms"]
          + row["render_ms"]
        )

    _append_row(path, row)

    next_index = combo_index + 1
    if next_index >= len(combos):
        return None    # signal “done”
    else:
        return next_index
    
def write_quality_simulation_csv(
    quality_buffer: deque,
    combos: List[Dict],
    combo_index: int,
    mode: EncodingMode,
):
    """
    Once qual_buffer fills up, average it and append a row with ONLY
    mean_pos_error, max_pos_error, mean_color_error into the per-mode CSV.
    Returns next combo_index, or None if no more combos.
    """

    if len(quality_buffer) < quality_buffer.maxlen:
        return combo_index

    avg = pd.DataFrame(quality_buffer).mean()
    quality_buffer.clear()

    if mode == EncodingMode.FULL:
        path = QUAL_FULL_CSV
    elif mode == EncodingMode.IMPORTANCE:
        path = QUAL_IMP_CSV
    else:
        return None  # skip NONE

    row = {
        "experiment_id":     combo_index,
        "mean_pos_error":   avg.get("mean_pos_error",   pd.NA),
        "max_pos_error":    avg.get("max_pos_error",    pd.NA),
        "mean_color_error": avg.get("mean_color_error", pd.NA),
        "mean_relative_pos_error(%)":   avg.get("mean_relative_pos_error(%)",   pd.NA),
        "max_relative_pos_error(%)":    avg.get("max_relative_pos_error(%)",    pd.NA),
        "mean_relative_color_error(%)": avg.get("mean_relative_color_error(%)", pd.NA),
    }

    _append_row(path, row)

    next_index = combo_index + 1
    if next_index >= len(combos):
        return None    # signal “done”
    else:
        return next_index