import csv
import struct
import time
from pathlib import Path

import cupy as cp
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLOE

import producer_cli as producer_cli
import sam2_camera_predictor as sam2_camera
from cuda_quantizer import CudaQuantizer, EncodingMode


def read_ply_ascii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a binary little-endian PLY with x y z r g b properties."""
    # The file header is ASCII, but the vertex block is binary.
    with open(path, "rb") as handle:
        header_lines = []
        line = handle.readline()
        while line:
            header_line = line.decode("ascii", errors="ignore").strip()
            header_lines.append(header_line)
            if header_line == "end_header":
                break
            line = handle.readline()

        vertex_count = 0
        properties: list[tuple[str, str]] = []
        in_vertex_block = False
        for entry in header_lines:
            if entry.startswith("element vertex"):
                vertex_count = int(entry.split()[-1])
                in_vertex_block = True
                continue
            if entry.startswith("element") and not entry.startswith("element vertex"):
                in_vertex_block = False
            if in_vertex_block and entry.startswith("property"):
                _, prop_type, prop_name = entry.split()
                properties.append((prop_type, prop_name))

        type_map = {
            "float": "<f4",
            "float32": "<f4",
            "uchar": "u1",
            "uint8": "u1",
            "int": "<i4",
            "int32": "<i4",
        }
        dtype_fields = [(name, type_map[prop]) for prop, name in properties]
        ply_dtype = np.dtype(dtype_fields)
        # Read the binary vertex payload in one shot for speed.
        raw = handle.read(vertex_count * ply_dtype.itemsize)
        data = np.frombuffer(raw, dtype=ply_dtype, count=vertex_count)

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    if {"red", "green", "blue"}.issubset(data.dtype.names):
        rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
    else:
        rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
    return xyz, rgb


def write_ply_binary(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary little-endian PLY with x y z and uchar RGB."""
    # Use a structured array so the binary layout matches the header.
    dtype_fields = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    vertex_dtype = np.dtype(dtype_fields)
    vertex_data = np.empty(xyz.shape[0], dtype=vertex_dtype)
    vertex_data["x"] = xyz[:, 0]
    vertex_data["y"] = xyz[:, 1]
    vertex_data["z"] = xyz[:, 2]
    vertex_data["red"] = rgb[:, 0]
    vertex_data["green"] = rgb[:, 1]
    vertex_data["blue"] = rgb[:, 2]
    with open(path, "wb") as handle:
        handle.write(b"ply\n")
        handle.write(b"format binary_little_endian 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n".encode("ascii"))
        handle.write(b"property float x\n")
        handle.write(b"property float y\n")
        handle.write(b"property float z\n")
        handle.write(b"property uchar red\n")
        handle.write(b"property uchar green\n")
        handle.write(b"property uchar blue\n")
        handle.write(b"end_header\n")
        handle.write(vertex_data.tobytes())


def build_cluster_map(binary_mask: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    """Create a 0/1/2 cluster map from a binary mask."""
    # Normalize incoming logits/masks to (H, W) before interpolation.
    if binary_mask.ndim > 2:
        while binary_mask.ndim > 2:
            binary_mask = binary_mask.squeeze(0)
    if binary_mask.ndim == 1:
        binary_mask = binary_mask.reshape(target_shape)
    # Keep tensors in (N, C, H, W) so pooling/interpolation stay 2D.
    mask_in = binary_mask.float().unsqueeze(0).unsqueeze(0)
    mask1_t = F.interpolate(mask_in, size=target_shape, mode="nearest")
    pooled = F.max_pool2d(mask1_t, kernel_size=32, stride=1, ceil_mode=True)
    expanded = F.interpolate(pooled, size=target_shape, mode="nearest")
    m1 = mask1_t > 0.5
    m2 = (expanded > 0.5) & (~m1)

    final_map = torch.zeros_like(m1, dtype=torch.uint8)
    final_map[m1] = 1
    final_map[m2] = 2
    # Drop batch/channel axes right before returning the map.
    return final_map.squeeze(0).squeeze(0)


def decode_chunk(buffer_bytes: bytes) -> tuple[np.ndarray, np.ndarray, int]:
    """Decode a quantized chunk into float XYZ and uint8 RGB."""
    header = struct.unpack("<6f2i", buffer_bytes[:32])
    min_vals = np.array(header[0:3], dtype=np.float32)
    scale_vals = np.array(header[3:6], dtype=np.float32)
    mode = header[6]
    num_points = header[7]

    offset = 32
    if mode == EncodingMode.HIGH.value:
        packed = np.frombuffer(buffer_bytes, dtype=np.uint32, count=num_points, offset=offset)
        offset += num_points * 4
        r = np.frombuffer(buffer_bytes, dtype=np.uint8, count=num_points, offset=offset)
        offset += num_points
        offset += (4 - (offset % 4)) % 4
        g = np.frombuffer(buffer_bytes, dtype=np.uint8, count=num_points, offset=offset)
        offset += num_points
        offset += (4 - (offset % 4)) % 4
        b = np.frombuffer(buffer_bytes, dtype=np.uint8, count=num_points, offset=offset)

        x = (packed & 2047) / 2047.0
        y = ((packed >> 11) & 2047) / 2047.0
        z = ((packed >> 22) & 1023) / 1023.0
        norm = np.stack([x, y, z], axis=1).astype(np.float32)
        xyz = (norm / scale_vals) + min_vals
        rgb = np.stack([r, g, b], axis=1).astype(np.uint8)
        return xyz, rgb, mode

    sx = 2 if mode == EncodingMode.MED.value else 1
    # MED carries 11/11/10 coordinates while LOW carries 8/8/8 coordinates.
    scol = 4
    bits_col = (8, 8, 8)

    coord_dtype = np.uint16 if sx == 2 else np.uint8
    x = np.frombuffer(buffer_bytes, dtype=coord_dtype, count=num_points, offset=offset)
    offset += num_points * sx
    offset += (4 - (offset % 4)) % 4
    y = np.frombuffer(buffer_bytes, dtype=coord_dtype, count=num_points, offset=offset)
    offset += num_points * sx
    offset += (4 - (offset % 4)) % 4
    z = np.frombuffer(buffer_bytes, dtype=coord_dtype, count=num_points, offset=offset)
    offset += num_points * sx
    offset += (4 - (offset % 4)) % 4
    packed = np.frombuffer(buffer_bytes, dtype=np.uint32, count=num_points, offset=offset)

    max_coord = np.array([
        float((1 << (11 if mode == EncodingMode.MED.value else 8)) - 1),
        float((1 << (11 if mode == EncodingMode.MED.value else 8)) - 1),
        float((1 << (10 if mode == EncodingMode.MED.value else 8)) - 1),
    ], dtype=np.float32)
    norm = np.stack([x, y, z], axis=1).astype(np.float32) / max_coord
    xyz = (norm / scale_vals) + min_vals

    br, bg, bb = bits_col
    packed_vals = packed.astype(np.uint32)
    r = packed_vals & ((1 << br) - 1)
    g = (packed_vals >> br) & ((1 << bg) - 1)
    b = (packed_vals >> (br + bg)) & ((1 << bb) - 1)
    r = (r.astype(np.float32) / ((1 << br) - 1) * 255.0).astype(np.uint8)
    g = (g.astype(np.float32) / ((1 << bg) - 1) * 255.0).astype(np.uint8)
    b = (b.astype(np.float32) / ((1 << bb) - 1) * 255.0).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=1)
    return xyz, rgb, mode


def allocate_budgets(
    n_in: int,
    n_mid: int,
    n_out: int,
    total_budget: int,
    min_keep_ratio_high: float,
    min_keep_ratio_med: float,
    min_keep_ratio_low: float,
) -> tuple[int, int, int]:
    """Split a point budget into in/mid/out buckets with per-cluster keep floors."""
    # Keep at least a configurable percentage from each cluster when possible.
    min_in = min(n_in, int(np.ceil(n_in * min_keep_ratio_high)))
    min_mid = min(n_mid, int(np.ceil(n_mid * min_keep_ratio_med)))
    min_out = min(n_out, int(np.ceil(n_out * min_keep_ratio_low)))

    min_required = min_in + min_mid + min_out
    if min_required >= total_budget:
        # If budget is too small, reduce floors proportionally and distribute leftovers.
        scale = total_budget / min_required if min_required > 0 else 0.0
        alloc_in = min(min_in, int(np.floor(min_in * scale)))
        alloc_mid = min(min_mid, int(np.floor(min_mid * scale)))
        alloc_out = min(min_out, int(np.floor(min_out * scale)))
        used = alloc_in + alloc_mid + alloc_out
        remaining = total_budget - used

        for cluster in ("in", "mid", "out"):
            if remaining <= 0:
                break
            if cluster == "in":
                extra = min(remaining, n_in - alloc_in)
                alloc_in += extra
            elif cluster == "mid":
                extra = min(remaining, n_mid - alloc_mid)
                alloc_mid += extra
            else:
                extra = min(remaining, n_out - alloc_out)
                alloc_out += extra
            remaining -= extra
        return alloc_in, alloc_mid, alloc_out

    budget_in = min_in
    budget_mid = min_mid
    budget_out = min_out
    remaining = total_budget - min_required

    # Fill remaining budget by importance order.
    add_in = min(remaining, n_in - budget_in)
    budget_in += add_in
    remaining -= add_in

    add_mid = min(remaining, n_mid - budget_mid)
    budget_mid += add_mid
    remaining -= add_mid

    add_out = min(remaining, n_out - budget_out)
    budget_out += add_out
    return budget_in, budget_mid, budget_out

def estimate_compressed_bytes_for_budget(
    quantizer: CudaQuantizer,
    n_in: int,
    n_mid: int,
    n_out: int,
    total_budget: int,
    min_keep_ratio_high: float,
    min_keep_ratio_med: float,
    min_keep_ratio_low: float,
) -> int:
    """Estimate compressed bytes for a candidate total budget."""
    # This mirrors the same importance split used during actual encoding.
    budget_in, budget_mid, budget_out = allocate_budgets(
        n_in,
        n_mid,
        n_out,
        total_budget,
        min_keep_ratio_high=min_keep_ratio_high,
        min_keep_ratio_med=min_keep_ratio_med,
        min_keep_ratio_low=min_keep_ratio_low,
    )
    total_bytes = 0
    if budget_in > 0:
        total_bytes += quantizer.estimate_buffer_size(EncodingMode.HIGH, budget_in)
    if budget_mid > 0:
        total_bytes += quantizer.estimate_buffer_size(EncodingMode.MED, budget_mid)
    if budget_out > 0:
        total_bytes += quantizer.estimate_buffer_size(EncodingMode.LOW, budget_out)
    return total_bytes


def derive_offline_point_budget(
    quantizer: CudaQuantizer,
    n_in: int,
    n_mid: int,
    n_out: int,
    target_fps: float,
    bandwidth_mb_per_s: float,
    min_keep_ratio_high: float,
    min_keep_ratio_med: float,
    min_keep_ratio_low: float,
) -> tuple[int, int]:
    """Compute the max point budget that fits frame bandwidth constraints."""
    # Convert MB/s to bytes per frame for the requested streaming fps.
    bytes_per_frame = int((bandwidth_mb_per_s * 1024.0 * 1024.0) / max(target_fps, 1e-6))
    max_points = n_in + n_mid + n_out
    if max_points <= 0:
        return 0, bytes_per_frame

    # Binary search finds the largest budget that stays within frame payload.
    left, right = 0, max_points
    best = 0
    while left <= right:
        mid = (left + right) // 2
        estimated_bytes = estimate_compressed_bytes_for_budget(
            quantizer,
            n_in,
            n_mid,
            n_out,
            mid,
            min_keep_ratio_high=min_keep_ratio_high,
            min_keep_ratio_med=min_keep_ratio_med,
            min_keep_ratio_low=min_keep_ratio_low,
        )
        if estimated_bytes <= bytes_per_frame:
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best, bytes_per_frame

def fast_subsample(d_indices: cp.ndarray, budget: int) -> cp.ndarray:
    """Pick evenly spaced indices on the GPU."""
    n = d_indices.size
    if n <= budget:
        return d_indices
    idx_to_keep = cp.linspace(0, n - 1, num=budget, dtype=cp.int32)
    return d_indices[idx_to_keep]


def write_depth_png(path: Path, depth_buf: np.ndarray) -> None:
    """Write depth to a 16-bit PNG after min/max normalization."""
    # Flattened buffers become a single-row image so they still serialize.
    if depth_buf.ndim == 1:
        depth_buf = depth_buf.reshape(1, -1)
    depth_min = float(depth_buf.min()) if depth_buf.size > 0 else 0.0
    depth_max = float(depth_buf.max()) if depth_buf.size > 0 else 1.0
    scale = 65535.0 / (depth_max - depth_min) if depth_max > depth_min else 1.0
    depth_u16 = np.clip((depth_buf - depth_min) * scale, 0.0, 65535.0).astype(np.uint16)
    cv2.imwrite(str(path), depth_u16)


def write_rgb_png(path: Path, rgb_buf: np.ndarray) -> None:
    """Write RGB data to PNG, reshaping single-row buffers if needed."""
    # Flattened buffers become a single-row image so they still serialize.
    if rgb_buf.ndim == 2:
        rgb_buf = rgb_buf.reshape(1, -1, 3)
    rgb_u8 = rgb_buf.astype(np.uint8, copy=False)
    cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


def rebuild_image_buffers(
    xyz_points: np.ndarray,
    rgb_points: np.ndarray,
    point_indices: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter decoded point buffers back to camera-sized depth and RGB images."""
    height, width = image_shape
    # Keep a dense camera grid so PNGs are always 2D images.
    depth_img = np.zeros((height, width), dtype=np.float32)
    rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
    if xyz_points.size == 0 or point_indices.size == 0:
        return depth_img, rgb_img

    # Map 1D point indices into 2D pixel coordinates.
    rows = point_indices // width
    cols = point_indices % width
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    depth_img[rows[valid], cols[valid]] = xyz_points[valid, 2]
    rgb_img[rows[valid], cols[valid]] = rgb_points[valid]
    return depth_img, rgb_img


def write_roi_debug_overlay(path: Path, color_img: np.ndarray, roi_box: tuple[int, int, int, int], label: str) -> None:
    """Write a debug overlay with the ROI rectangle and label."""
    # Copy before drawing so the original frame stays untouched.
    overlay = color_img.copy()
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(overlay, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def bytes_to_mb(num_bytes: int) -> float:
    """Convert raw byte counts to MiB for CSV reporting."""
    # Keep this conversion centralized so CSV units stay consistent.
    return float(num_bytes) / (1024.0 * 1024.0)


def load_frame_rgb(input_root: Path, frame_stem: str, fallback_rgb: np.ndarray) -> np.ndarray:
    """Load an exported RGB frame PNG for clustering, falling back to PLY colors."""
    # Exported frames follow frame_XXXX_rgb.png, keyed by the PLY stem.
    png_path = input_root / f"{frame_stem}_rgb.png"
    bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if bgr is None:
        # Fall back to reconstructed colors when the PNG is missing.
        return fallback_rgb
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth_valid_indices(input_root: Path, frame_stem: str, depth_shape: tuple[int, int]) -> np.ndarray | None:
    """Load flat valid-depth pixel indices for a frame if a depth PNG exists."""
    # Offline exports can carry frame_XXXX_depth.png next to frame_XXXX.ply.
    depth_path = input_root / f"{frame_stem}_depth.png"
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return None
    if depth_img.ndim == 3:
        depth_img = depth_img[..., 0]
    if tuple(depth_img.shape[:2]) != depth_shape:
        return None
    # PLY generation keeps depth>0 points, so use the same mask ordering.
    return np.flatnonzero(depth_img.reshape(-1) > 0).astype(np.int32, copy=False)


def run_offline_compression(args, server=None) -> None:
    """Compress all PLY files in a folder, decode outputs, and optionally stream chunks."""
    # Offline input/output are anchored under producer/exported_PCs.
    export_root = Path(__file__).resolve().parent / "exported_PCs"
    # The offline prefix names the input folder and the output folder suffix.
    input_root = export_root / args.offline_prefix
    output_root = export_root / f"{args.offline_prefix}_IMPORTANCE"
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[offline] input: {input_root} output: {output_root}")

    # Bind the GPU once so CuPy ops stay on the same device.
    cp.cuda.Device(0).use()
    cmr_clr_width, cmr_clr_height = producer_cli.map_to_camera_res[args.realsense_clr_stream]
    cmr_depth_width, cmr_depth_height = producer_cli.map_to_camera_res[args.realsense_depth_stream]

    quantizer = CudaQuantizer()
    stream_in = cp.cuda.Stream(non_blocking=True)
    stream_med = cp.cuda.Stream(non_blocking=True)
    stream_out = cp.cuda.Stream(non_blocking=True)

    
# Allocate pinned buffers for the worst-case frame budget at the selected depth resolution.
    max_points_per_frame = cmr_depth_width * cmr_depth_height
    pinned_mem_high = cp.cuda.alloc_pinned_memory(max_points_per_frame * 15)
    pinned_mem_med = cp.cuda.alloc_pinned_memory(max_points_per_frame * 15)
    pinned_mem_low = cp.cuda.alloc_pinned_memory(max_points_per_frame * 15)

    pinned_np_high = np.frombuffer(pinned_mem_high, dtype=np.uint8)
    pinned_np_med = np.frombuffer(pinned_mem_med, dtype=np.uint8)
    pinned_np_low = np.frombuffer(pinned_mem_low, dtype=np.uint8)

    csv_path = output_root / "compression_report.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "method",
            "points_full",
            "points_cluster_high",
            "points_cluster_mid",
            "points_cluster_low",
            "uncompressed_size_mb",
            "cluster_high_size_mb",
            "cluster_mid_size_mb",
            "cluster_low_size_mb",
            "total_compressed_size_mb",
            "detection_time_ms",
            "encode_time_ms",
            "point_budget",
            "keep_ratio",
            "keep_ratio_high",
            "keep_ratio_mid",
            "keep_ratio_low",
        ]
    )

    predictor = None
    yolo_class_idx = 0
    roi_init = False
    prev_cluster = None

    if args.cluster_predictor == "yolo":
        model_path = "yoloe-11l-seg.pt" if args.yolo_size == "large" else "yoloe-11s-seg.pt"
        predictor = YOLOE(model_path, verbose=False)
        names = ["person"]
        predictor.set_classes(names, predictor.get_text_pe(names))

    if args.cluster_predictor == "sam2":
        enum = producer_cli.map_to_enum[args.sam2_checkpoint]
        link = producer_cli.map_to_config[enum]
        path_to_yaml = Path(producer_cli.CONFIG_PATH, link[0])
        path_to_chkp = Path(producer_cli.CHECKPOINT_PATH, Path(link[1]).name)
        if not path_to_chkp.exists():
            producer_cli.getRequest(producer_cli.CHECKPOINT_PATH, link[1])
        predictor = sam2_camera.build_sam2_camera_predictor(
            config_file=path_to_yaml.name,
            config_path=str(Path(".", "configs", "sam2.1")),
            ckpt_path=str(path_to_chkp),
            device=args.device,
            image_size=args.sam2_image_size,
        )

    ply_files = sorted(input_root.glob("*.ply"))
    for frame_idx, ply_path in enumerate(ply_files):
        print(f"[offline] frame {frame_idx + 1}/{len(ply_files)}: loading {ply_path.name}")
        xyz, rgb = read_ply_ascii(ply_path)
        d_vertices = cp.asarray(xyz, dtype=cp.float32)
        d_colors = cp.asarray(rgb, dtype=cp.uint8)

        color_img = np.zeros((cmr_clr_height, cmr_clr_width, 3), dtype=np.uint8)
        if xyz.shape[0] == cmr_clr_width * cmr_clr_height:
            color_img = rgb.reshape(cmr_clr_height, cmr_clr_width, 3)
        # Prefer the exported RGB frame for clustering, with a PLY-derived fallback.
        color_img = load_frame_rgb(input_root, ply_path.stem, color_img)

        detection_ms = 0.0
        if predictor is not None and args.cluster_predictor == "yolo":
            # YOLO keeps per-frame segmentation in offline mode.
            print(f"[offline] running {args.cluster_predictor} segmentation on {ply_path.name}")
            detect_start = time.perf_counter()
            if args.cluster_predictor == "yolo":
                # Offline YOLO combines masks per class, so log that no ROI is used.
                if args.offline_debug_roi:
                    print("[offline][roi] yolo uses combined masks for the target class (no explicit ROI).")
                result = predictor.predict(color_img, conf=0.1, verbose=False)[0]
                # Keep the mask accumulation on torch so shape fixes can stay on GPU.
                mask = torch.zeros((cmr_clr_height, cmr_clr_width), dtype=torch.bool, device=args.device)
                if result.masks is not None:
                    classes = result.boxes.cls.cpu().numpy().astype(np.int32)
                    for box_i, cls_i in enumerate(classes):
                        if cls_i != yolo_class_idx:
                            continue
                        yolo_mask = result.masks.data[box_i] > 0.5
                        # YOLO can output a fixed inference shape, so remap to camera resolution.
                        if tuple(yolo_mask.shape[-2:]) != (cmr_clr_height, cmr_clr_width):
                            yolo_mask = F.interpolate(
                                yolo_mask.unsqueeze(0).unsqueeze(0).float(),
                                size=(cmr_clr_height, cmr_clr_width),
                                mode="nearest",
                            ).squeeze(0).squeeze(0) > 0.5
                        mask |= yolo_mask
                    if args.offline_debug_roi:
                        # Draw detected boxes so the ROI-free selection is visible.
                        yolo_debug = color_img.copy()
                        for box in result.boxes.xyxy.cpu().numpy().astype(np.int32):
                            x1, y1, x2, y2 = box.tolist()
                            cv2.rectangle(yolo_debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        debug_path = output_root / f"{ply_path.stem}_yolo_roi_debug.png"
                        cv2.imwrite(str(debug_path), cv2.cvtColor(yolo_debug, cv2.COLOR_RGB2BGR))
                cluster_map = build_cluster_map(mask, (cmr_clr_height, cmr_clr_width))
                prev_cluster = cluster_map
            detection_ms = (time.perf_counter() - detect_start) * 1000.0
        elif predictor is not None and args.cluster_predictor == "sam2" and not roi_init:
            # SAM2 initializes once from the CLI ROI and then switches to tracking.
            print(f"[offline] initializing {args.cluster_predictor} ROI on {ply_path.name}")
            detect_start = time.perf_counter()
            if args.offline_debug_roi:
                print(
                    f"[offline][roi] sam2 prompt center=({args.offline_query_x:.1f}, {args.offline_query_y:.1f}) "
                    f"box={args.offline_box_size:.1f}px"
                )
                roi_box = (
                    int(args.offline_query_x - args.offline_box_size),
                    int(args.offline_query_y - args.offline_box_size),
                    int(args.offline_query_x + args.offline_box_size),
                    int(args.offline_query_y + args.offline_box_size),
                )
                overlay_path = output_root / f"{ply_path.stem}_sam2_roi_debug.png"
                write_roi_debug_overlay(overlay_path, color_img, roi_box, "SAM2 ROI")
            roi_center = torch.tensor(
                [
                    [args.offline_query_x, args.offline_query_y],
                    [args.offline_query_x - args.offline_box_size, args.offline_query_y - args.offline_box_size],
                    [args.offline_query_x - args.offline_box_size, args.offline_query_y + args.offline_box_size],
                    [args.offline_query_x + args.offline_box_size, args.offline_query_y - args.offline_box_size],
                    [args.offline_query_x + args.offline_box_size, args.offline_query_y + args.offline_box_size],
                ],
                dtype=torch.float32,
                device=args.device,
            )
            predictor.load_first_frame(torch.as_tensor(color_img, device=args.device).permute(2, 0, 1))
            _, _, out_mask_logits = predictor.add_new_prompt(
                frame_idx=0,
                obj_id=(1,),
                points=roi_center,
                labels=np.array([1, 1, 1, 1, 1], dtype=np.int32),
            )
            mask = out_mask_logits[0] > 0.0
            prev_cluster = build_cluster_map(mask, (cmr_clr_height, cmr_clr_width))
            roi_init = True
            detection_ms = (time.perf_counter() - detect_start) * 1000.0
        elif predictor is not None and args.cluster_predictor == "sam2" and roi_init:
            # After init, every frame uses SAM2 tracking only.
            print(f"[offline] tracking {args.cluster_predictor} mask on {ply_path.name}")
            detect_start = time.perf_counter()
            _, out_mask_logits = predictor.track(torch.as_tensor(color_img, device=args.device).permute(2, 0, 1))
            mask = out_mask_logits[0] > 0.0
            prev_cluster = build_cluster_map(mask, (cmr_clr_height, cmr_clr_width))
            detection_ms = (time.perf_counter() - detect_start) * 1000.0

        if prev_cluster is None:
            prev_cluster = torch.zeros((cmr_clr_height, cmr_clr_width), dtype=torch.uint8, device=args.device)

        if tuple(prev_cluster.shape[-2:]) != (cmr_depth_height, cmr_depth_width):
            # Segmentation runs on RGB, but PLY ordering follows depth pixels.
            cluster_for_points = F.interpolate(
                prev_cluster.float().unsqueeze(0).unsqueeze(0),
                size=(cmr_depth_height, cmr_depth_width),
                mode="nearest",
            ).squeeze(0).squeeze(0).to(torch.uint8)
        else:
            cluster_for_points = prev_cluster

        flat_indices_cpu = load_depth_valid_indices(
            input_root,
            ply_path.stem,
            (cmr_depth_height, cmr_depth_width),
        )
        if flat_indices_cpu is None:
            if d_vertices.shape[0] == cmr_depth_width * cmr_depth_height:
                flat_indices_cpu = np.arange(d_vertices.shape[0], dtype=np.int32)
            else:
                flat_indices_cpu = np.arange(d_vertices.shape[0], dtype=np.int32)
                print(
                    f"[offline][warn] missing {ply_path.stem}_depth.png; using vertex-order indices for rasterization"
                )

        if flat_indices_cpu.shape[0] != d_vertices.shape[0]:
            print(
                f"[offline][warn] depth index count mismatch on {ply_path.name}; using vertex-order indices"
            )
            flat_indices_cpu = np.arange(d_vertices.shape[0], dtype=np.int32)

        cluster_flat = cluster_for_points.reshape(-1).cpu().numpy()
        point_cluster_cpu = cluster_flat[flat_indices_cpu]
        d_cluster_values = cp.asarray(point_cluster_cpu, dtype=cp.uint8)
        d_point_idx = cp.arange(d_vertices.shape[0], dtype=cp.int32)
        d_flat_idx = cp.asarray(flat_indices_cpu, dtype=cp.int32)

        d_in_mask = d_cluster_values == 1
        d_mid_mask = d_cluster_values == 2
        d_out_mask = d_cluster_values == 0

        n_in = int(cp.count_nonzero(d_in_mask))
        n_mid = int(cp.count_nonzero(d_mid_mask))
        n_out = int(cp.count_nonzero(d_out_mask))

        # Derive this frame budget from target fps and available bandwidth.
        derived_budget, target_frame_bytes = derive_offline_point_budget(
            quantizer,
            n_in,
            n_mid,
            n_out,
            args.offline_target_fps,
            args.offline_bandwidth_mb_per_s,
            args.min_keep_ratio_high,
            args.min_keep_ratio_med,
            args.min_keep_ratio_low,
        )
        total_budget = derived_budget
        budget_in, budget_mid, budget_out = allocate_budgets(
            n_in,
            n_mid,
            n_out,
            total_budget,
            args.min_keep_ratio_high,
            args.min_keep_ratio_med,
            args.min_keep_ratio_low,
        )
        estimated_payload = estimate_compressed_bytes_for_budget(
            quantizer,
            n_in,
            n_mid,
            n_out,
            total_budget,
            args.min_keep_ratio_high,
            args.min_keep_ratio_med,
            args.min_keep_ratio_low,
        )
        print(
            f"[offline] budgets for {ply_path.name}: in={budget_in} mid={budget_mid} out={budget_out} "
            f"(total={total_budget}, est={estimated_payload}B, target={target_frame_bytes}B @ {args.offline_target_fps:.2f}fps)"
        )


        # This mirrors online framing so the web client can process chunks in real-time order.
        frame_id_byte = frame_idx % 255
        byte_offset = 0

        broadcast_buffers = {}
        with stream_in:
            if budget_in > 0:
                d_in_point_idx = d_point_idx[d_in_mask]
                d_in_flat_idx = d_flat_idx[d_in_mask]
                if d_in_point_idx.size > budget_in:
                    keep_pos = fast_subsample(cp.arange(d_in_point_idx.size, dtype=cp.int32), budget_in)
                    d_in_point_idx = d_in_point_idx[keep_pos]
                    d_in_flat_idx = d_in_flat_idx[keep_pos]
                if d_in_point_idx.size > 0:
                    print(f"[offline] encoding HIGH for {ply_path.name} ({int(d_in_point_idx.size)} points)")
                    t_start = time.perf_counter()
                    res_view = quantizer.encode(
                        stream_in,
                        EncodingMode.HIGH,
                        d_vertices[d_in_point_idx],
                        d_colors[d_in_point_idx],
                        pinned_np_high,
                    )
                    stream_in.synchronize()
                    t_ms = (time.perf_counter() - t_start) * 1000.0
                    # Track source depth pixels so raster output matches frame geometry.
                    idx_cpu = cp.asnumpy(d_in_flat_idx).astype(np.int32, copy=False)
                    broadcast_buffers[EncodingMode.HIGH] = (d_in_point_idx.size, res_view, t_ms, idx_cpu)

        with stream_med:
            if budget_mid > 0:
                d_mid_point_idx = d_point_idx[d_mid_mask]
                d_mid_flat_idx = d_flat_idx[d_mid_mask]
                if d_mid_point_idx.size > budget_mid:
                    keep_pos = fast_subsample(cp.arange(d_mid_point_idx.size, dtype=cp.int32), budget_mid)
                    d_mid_point_idx = d_mid_point_idx[keep_pos]
                    d_mid_flat_idx = d_mid_flat_idx[keep_pos]
                if d_mid_point_idx.size > 0:
                    print(f"[offline] encoding MED for {ply_path.name} ({int(d_mid_point_idx.size)} points)")
                    t_start = time.perf_counter()
                    res_view = quantizer.encode(
                        stream_med,
                        EncodingMode.MED,
                        d_vertices[d_mid_point_idx],
                        d_colors[d_mid_point_idx],
                        pinned_np_med,
                    )
                    stream_med.synchronize()
                    t_ms = (time.perf_counter() - t_start) * 1000.0
                    idx_cpu = cp.asnumpy(d_mid_flat_idx).astype(np.int32, copy=False)
                    broadcast_buffers[EncodingMode.MED] = (d_mid_point_idx.size, res_view, t_ms, idx_cpu)

        with stream_out:
            if budget_out > 0:
                d_out_point_idx = d_point_idx[d_out_mask]
                d_out_flat_idx = d_flat_idx[d_out_mask]
                if d_out_point_idx.size > budget_out:
                    keep_pos = fast_subsample(cp.arange(d_out_point_idx.size, dtype=cp.int32), budget_out)
                    d_out_point_idx = d_out_point_idx[keep_pos]
                    d_out_flat_idx = d_out_flat_idx[keep_pos]
                if d_out_point_idx.size > 0:
                    print(f"[offline] encoding LOW for {ply_path.name} ({int(d_out_point_idx.size)} points)")
                    t_start = time.perf_counter()
                    res_view = quantizer.encode(
                        stream_out,
                        EncodingMode.LOW,
                        d_vertices[d_out_point_idx],
                        d_colors[d_out_point_idx],
                        pinned_np_low,
                    )
                    stream_out.synchronize()
                    t_ms = (time.perf_counter() - t_start) * 1000.0
                    idx_cpu = cp.asnumpy(d_out_flat_idx).astype(np.int32, copy=False)
                    broadcast_buffers[EncodingMode.LOW] = (d_out_point_idx.size, res_view, t_ms, idx_cpu)

        # Use encoded chunk count, not budget count, so frame completion is accurate on client.
        num_chunks = len(broadcast_buffers)

        assembled_xyz: list[np.ndarray] = []
        assembled_rgb: list[np.ndarray] = []
        assembled_idx: list[np.ndarray] = []
        assembled_buffers: list[bytes] = []
        total_encode_ms = 0.0
        total_kept_points = 0
        encoded_size_by_mode = {
            EncodingMode.HIGH: 0,
            EncodingMode.MED: 0,
            EncodingMode.LOW: 0,
        }
        # Track per-cluster kept points after budget clipping and subsampling.
        kept_points_by_mode = {
            EncodingMode.HIGH: 0,
            EncodingMode.MED: 0,
            EncodingMode.LOW: 0,
        }

        for mode in (EncodingMode.HIGH, EncodingMode.MED, EncodingMode.LOW):
            if mode not in broadcast_buffers:
                continue
            count, buffer_view, t_ms, idx_cpu = broadcast_buffers[mode]
            buffer_bytes = buffer_view.tobytes()
            mode_name = mode.name.lower()

            # Stream each completed chunk right away so the client can decode and render progressively.
            if server is not None and num_chunks > 0:
                header = (
                    num_chunks.to_bytes(1, "little")
                    + frame_id_byte.to_bytes(1, "little")
                    + byte_offset.to_bytes(4, "little")
                )
                server.broadcast(header + buffer_bytes)
                byte_offset += len(buffer_bytes)

            print(f"[offline] decoding {mode_name} for {ply_path.name} ({int(count)} points)")
            # Decode each importance tier, then assemble into a single output.
            xyz_dec, rgb_dec, _ = decode_chunk(buffer_bytes)
            assembled_xyz.append(xyz_dec)
            assembled_rgb.append(rgb_dec)
            assembled_idx.append(idx_cpu)
            assembled_buffers.append(buffer_bytes)
            total_encode_ms += t_ms
            total_kept_points += int(count)
            kept_points_by_mode[mode] = int(count)
            # Track encoded payload size for each cluster tier.
            encoded_size_by_mode[mode] = len(buffer_bytes)

        if assembled_xyz:
            # Sort by original pixel index so PLY and PNG outputs share the same point order.
            xyz_all = np.concatenate(assembled_xyz, axis=0)
            rgb_all = np.concatenate(assembled_rgb, axis=0)
            idx_all = np.concatenate(assembled_idx, axis=0)
            order = np.argsort(idx_all)
            xyz_all = xyz_all[order]
            rgb_all = rgb_all[order]
            idx_all = idx_all[order]
            buffer_all = b"".join(assembled_buffers)
        else:
            # Fall back to empty outputs if no clusters were encoded.
            xyz_all = np.zeros((0, 3), dtype=np.float32)
            rgb_all = np.zeros((0, 3), dtype=np.uint8)
            idx_all = np.zeros((0,), dtype=np.int32)
            buffer_all = b""

        out_bin = output_root / f"{ply_path.stem}.bin"
        out_bin.write_bytes(buffer_all)

        out_ply = output_root / f"{ply_path.stem}.ply"
        write_ply_binary(out_ply, xyz_all, rgb_all)

        depth_path = output_root / f"{ply_path.stem}_depth.png"
        rgb_path = output_root / f"{ply_path.stem}_rgb.png"
        depth_buf, rgb_buf = rebuild_image_buffers(
            xyz_all,
            rgb_all,
            idx_all,
            (cmr_depth_height, cmr_depth_width),
        )
        write_depth_png(depth_path, depth_buf)
        write_rgb_png(rgb_path, rgb_buf)

        original_points = int(xyz.shape[0])
        original_size_mb = bytes_to_mb(xyz.nbytes + rgb.nbytes)
        high_size_mb = bytes_to_mb(encoded_size_by_mode[EncodingMode.HIGH])
        mid_size_mb = bytes_to_mb(encoded_size_by_mode[EncodingMode.MED])
        low_size_mb = bytes_to_mb(encoded_size_by_mode[EncodingMode.LOW])
        total_cluster_size_mb = high_size_mb + mid_size_mb + low_size_mb
        keep_ratio = float(total_kept_points) / original_points if original_points > 0 else 0.0
        # Each cluster ratio compares kept points against cluster source points.
        keep_ratio_high = float(kept_points_by_mode[EncodingMode.HIGH]) / n_in if n_in > 0 else 0.0
        keep_ratio_mid = float(kept_points_by_mode[EncodingMode.MED]) / n_mid if n_mid > 0 else 0.0
        keep_ratio_low = float(kept_points_by_mode[EncodingMode.LOW]) / n_out if n_out > 0 else 0.0
        # Log full frame stats plus per-cluster point and buffer metrics.
        csv_writer.writerow(
            [
                "Cuda quantization",
                original_points,
                n_in,
                n_mid,
                n_out,
                f"{original_size_mb:.6f}",
                f"{high_size_mb:.6f}",
                f"{mid_size_mb:.6f}",
                f"{low_size_mb:.6f}",
                f"{total_cluster_size_mb:.6f}",
                f"{detection_ms:.3f}",
                f"{total_encode_ms:.3f}",
                total_budget,
                f"{keep_ratio:.6f}",
                f"{keep_ratio_high:.6f}",
                f"{keep_ratio_mid:.6f}",
                f"{keep_ratio_low:.6f}",
            ]
        )

    csv_file.close()
