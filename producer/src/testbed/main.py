import argparse
import glob
import csv
import os

from pathlib import Path
from typing import List

from draco_wrapper import DracoWrapper
from .file_bencher import FileBencher, Metrics


def _expand_inputs(patterns: List[str | Path]) -> List[Path]:
    files: List[Path] = []

    for pat in patterns:
        pat_str = os.path.expanduser(str(pat))
        p = Path(pat_str)

        # If it's a glob pattern (supports absolute/relative and **)
        if any(ch in pat_str for ch in "*?[]"):
            files.extend(Path(m) for m in glob.glob(pat_str, recursive=True))

        # Directory: take .ply files directly inside (adjust to rglob if you want recursion)
        elif p.is_dir():
            files.extend(p.glob("*.ply"))

        # Otherwise it's a single path
        else:
            files.append(p)

    # Keep only existing .ply files (case-insensitive) and dedupe while preserving order
    seen = set()
    out: List[Path] = []
    for f in files:
        try:
            f = f.resolve()
        except Exception:
            pass
        if f.exists() and f.suffix.lower() == ".ply" and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def run(
    inputs: List[str | Path],
    repeats: int,
    qpos: int,
    qcol: int,
    speed_encode: int,
    speed_decode: int,
    warmup: bool = False,
    drop_first: bool = False,
) -> List[Metrics]:

    files = _expand_inputs(inputs)
    print(f"Benchmarking {len(files)} PLY files…")

    if not files:
        print("No .ply files found.")
        return []

    wrapper = DracoWrapper(
        position_quantization_bits=qpos,
        color_quantization_bits=qcol,
        speed_encode=speed_encode,
        speed_decode=speed_decode,
    )

    bencher = FileBencher(warmup=warmup, drop_first=drop_first)

    rows: List[Metrics] = []
    for file in files:
        metrics_instance: Metrics = bencher.bench_file(file, wrapper, repeats=repeats)
        rows.append(metrics_instance)
        print(
            f"{file.name:30s} "
            f"pts:{metrics_instance['points']:8d}  raw:{metrics_instance['raw_bytes']/1024:.1f} KiB  "
            f"enc~:{metrics_instance['encoded_bytes']/1024:.1f} KiB  "
            f"ratio:{(metrics_instance['ratio_raw_over_encoded'] or 0):5.2f}  "
            f"bpp:{(metrics_instance['bits_per_point'] or 0):5.2f}  "
            f"time:{metrics_instance['compression_ms_mean']:.1f}±{metrics_instance['compression_ms_std']:.1f} ms"
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description="Draco encode benchmark for PLY files")
    ap.add_argument("--inputs", nargs="+", required=True, help="PLY files or glob patterns (e.g. data/*.ply)")
    ap.add_argument("--repeats", type=int, default=3, help="Encode repeats per file (default: 3)")
    ap.add_argument("--qpos", type=int, default=11, help="Position quantization bits (default: 11)")
    ap.add_argument("--qcol", type=int, default=8, help="Color quantization bits (default: 8)")
    ap.add_argument("--speed-encode", type=int, default=10, help="Draco encode speed 0–10 (default: 10)")
    ap.add_argument("--speed-decode", type=int, default=10, help="Draco decode speed 0–10 (default: 10)")
    ap.add_argument("--warmup", action="store_true", help="Do a warmup encode before timing")
    ap.add_argument("--drop-first", action="store_true", help="Drop first timed run from stats")
    ap.add_argument("--csv", type=str, default=None, help="Write results to CSV")
    args = ap.parse_args()

    rows: List[Metrics] = run(
        inputs=args.inputs,
        repeats=args.repeats,
        qpos=args.qpos,
        qcol=args.qcol,
        speed_encode=args.speed_encode,
        speed_decode=args.speed_decode,
        warmup=args.warmup,
        drop_first=args.drop_first,
    )

    total_pts = sum(metrics_instance["points"] for metrics_instance in rows)
    total_raw = sum(metrics_instance["raw_bytes"] for metrics_instance in rows)
    total_enc = sum(metrics_instance["encoded_bytes"] for metrics_instance in rows)
    total_time_ms = sum(metrics_instance["compression_ms_mean"] for metrics_instance in rows)

    print("\n=== Summary ===")
    overall_ratio = (total_raw / total_enc) if total_enc > 0 else 0
    overall_bpp = (8.0 * total_enc / total_pts) if total_pts > 0 else 0
    print(
        f"Files: {len(rows)} | Points: {total_pts} | "
        f"Raw: {total_raw/1024:.1f} KiB | Encoded: {total_enc/1024:.1f} KiB | "
        f"Ratio: {overall_ratio:.2f} | bpp: {overall_bpp:.2f} | "
        f"Total encode time (mean-sum): {total_time_ms:.1f} ms"
    )

    if args.csv and rows:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        metrics_object_fieldnames = list(rows[0].as_row().keys())
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=metrics_object_fieldnames)
            w.writeheader()
            for m in rows:
                w.writerow(m.as_row())

        print(f"Wrote CSV: {out}")


if __name__ == "__main__":
    main()
