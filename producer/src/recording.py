
import csv
import time

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLOE

from draco_wrapper.draco_wrapper import DracoWrapper
from draco_wrapper import draco_bindings as dcb
from utils import write_pointcloud_ply
from sam2_camera_predictor import build_sam2_camera_predictor
from gesture_recognition import NormalizedBoundingBox


BYTES_PER_POINT = 15  # 3 float32 coordinates and 3 uint8 color channels.


def _append_stats(
    stats_rows: List[List[float]],
    ply_name: str,
    effective_ratio: float,
    encode_ms: float | None,
    decode_ms: float | None,
    deleted_points: int,
) -> None:
    stats_rows.append(
        [
            ply_name,
            round(effective_ratio, 6),
            None if encode_ms is None else round(encode_ms, 3),
            None if decode_ms is None else round(decode_ms, 3),
            deleted_points,
        ]
    )


def _write_stats_csv(
    stats_rows: List[List[float]],
    stats_path: Path,
    include_codec_metrics: bool = True,
    in_roi_pos_bits: int | None = None,
    out_roi_pos_bits: int | None = None,
    in_roi_color_bits: int | None = None,
    out_roi_color_bits: int | None = None,
) -> None:

    headers = ["ply_name", "effective_ratio"]
    if include_codec_metrics:
        headers.extend(["encode_time_ms", "decode_time_ms"])
    headers.append("points_deleted")

    quant_bit_pairs = [
        ("in_roi_pos_bits", in_roi_pos_bits),
        ("out_roi_pos_bits", out_roi_pos_bits),
        ("in_roi_color_bits", in_roi_color_bits),
        ("out_roi_color_bits", out_roi_color_bits),
    ]
    quant_headers = [name for name, value in quant_bit_pairs if value is not None]
    quant_values = [value for _, value in quant_bit_pairs if value is not None]
    if quant_headers:
        headers.extend(quant_headers)

    with open(stats_path, "w", newline="") as stats_file:
        writer = csv.writer(stats_file)
        writer.writerow(headers)

        if include_codec_metrics:
            for row in stats_rows:
                # Attach quantization bits so each frame log is self describing.
                row_to_write = row + quant_values
                writer.writerow(row_to_write)
            encode_values = [row[2] for row in stats_rows if row[2] is not None]
            decode_values = [row[3] for row in stats_rows if row[3] is not None]
            avg_encode = sum(encode_values) / len(encode_values) if encode_values else 0.0
            avg_decode = sum(decode_values) / len(decode_values) if decode_values else 0.0
            avg_ratio = sum(row[1] for row in stats_rows) / len(stats_rows)
            avg_deleted = sum(row[4] for row in stats_rows) / len(stats_rows)

            writer.writerow(
                [
                    "average",
                    round(avg_ratio, 6),
                    round(avg_encode, 3),
                    round(avg_decode, 3),
                    round(avg_deleted, 2),
                    *quant_values,
                ]
            )
        else:
            for row in stats_rows:
                row_to_write = [row[0], row[1], row[4], *quant_values]
                writer.writerow(row_to_write)

            avg_ratio = sum(row[1] for row in stats_rows) / len(stats_rows)
            avg_deleted = sum(row[4] for row in stats_rows) / len(stats_rows)

            writer.writerow(
                [
                    "average",
                    round(avg_ratio, 6),
                    round(avg_deleted, 2),
                    *quant_values,
                ]
            )


def compute_effective_subsample_ratio(
    valid_points: int,
    bandwidth_mbps: float,
    target_frame_rate: float,
    bytes_per_point: int = BYTES_PER_POINT,
) -> float:

    if valid_points <= 0 or target_frame_rate <= 0:
        return 1.0

    budget_bytes = (bandwidth_mbps * 1_000_000.0) / target_frame_rate
    point_budget = budget_bytes / float(bytes_per_point)

    ratio = point_budget / float(valid_points)
    return float(min(1.0, max(0.0, ratio)))


def _quantized_point_bytes(pos_bits: int, color_bits: int) -> float:
    bits_per_point = 3 * pos_bits + 3 * color_bits
    return bits_per_point / 8.0


def compute_importance_subsample_ratio(
    roi_points: int,
    non_roi_points: int,
    bandwidth_mbps: float,
    target_frame_rate: float,
    in_roi_pos_bits: int,
    in_roi_color_bits: int,
    out_roi_pos_bits: int,
    out_roi_color_bits: int,
) -> float:

    per_frame_budget_bytes = (bandwidth_mbps * 1_000_000.0) / target_frame_rate

    roi_bytes = float(roi_points) * _quantized_point_bytes(in_roi_pos_bits, in_roi_color_bits)
    remaining_budget = per_frame_budget_bytes - roi_bytes

    out_point_bytes = _quantized_point_bytes(out_roi_pos_bits, out_roi_color_bits)
    max_out_bytes = float(non_roi_points) * out_point_bytes

    ratio = remaining_budget / max_out_bytes
    return ratio


@dataclass
class RecordingManager:

    export_root: Path
    frame_target: int
    set_index: int = field(default=0)
    frames_remaining: int = field(default=0)
    current_dir: Optional[Path] = field(default=None)
    buffered_frames: List[Dict] = field(default_factory=list)
    is_recording: bool = field(default=False)
    frame_times: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        existing = [
            int(p.name.replace("PC_set", ""))
            for p in self.export_root.glob("PC_set*")
            if p.name.replace("PC_set", "").isdigit()
        ]
        self.set_index = max(existing, default=0)
        self.export_root.mkdir(parents=True, exist_ok=True)

    def start_recording(self) -> Path:
        highest = [
            int(p.name.replace("PC_set", ""))
            for p in self.export_root.glob("PC_set*")
            if p.name.replace("PC_set", "").isdigit()
        ]
        self.set_index = max(highest, default=self.set_index) + 1
        self.frames_remaining = self.frame_target
        self.current_dir = self.export_root / f"PC_set{self.set_index}"
        self.current_dir.mkdir(parents=True, exist_ok=True)
        self.buffered_frames.clear()
        self.frame_times.clear()
        self.is_recording = True
        return self.current_dir

    @property
    def is_active(self) -> bool:
        # Signal if a recording batch is collecting frames.
        return self.is_recording and self.frames_remaining > 0

    def capture_frame(
        self,
        vertices: np.ndarray,
        colors: np.ndarray,
        color_img: np.ndarray,
        depth_img: np.ndarray,
        valid_mask: np.ndarray,
        capture_time_ms: Optional[float] = None,
        record_native_color: bool = False,
    ) -> None:

        if not self.is_active:
            return

        valid_mask_image = valid_mask.reshape(color_img.shape[:2]).copy()

        frame_packet = {
            "vertices": vertices[valid_mask].copy(),
            "colors": colors[valid_mask].copy(),
            "color_img": color_img.copy(),
            "depth_img": depth_img.copy(),
            "valid_mask_image": valid_mask_image,
        }

        if record_native_color:
            # Preserve the unmodified RGB buffer for native-resolution exports.
            frame_packet["native_color_img"] = color_img.copy()

        if capture_time_ms is not None:
            self.frame_times.append(capture_time_ms)

        self.buffered_frames.append(frame_packet)
        self.frames_remaining -= 1

        if self.frames_remaining == 0:
            self.export_buffer()

    def export_buffer(self) -> None:
        if not self.buffered_frames:
            return

        if self.current_dir is None:
            self.current_dir = self.export_root / f"PC_set{self.set_index}"
            self.current_dir.mkdir(parents=True, exist_ok=True)

        for idx, packet in enumerate(self.buffered_frames, start=1):
            ply_path = self.current_dir / f"frame_{idx:04d}.ply"
            rgb_path = self.current_dir / f"frame_{idx:04d}_rgb.png"
            depth_path = self.current_dir / f"frame_{idx:04d}_depth.png"
            native_rgb_path = self.current_dir / f"frame_{idx:04d}_rgb_native.png"

            write_pointcloud_ply(ply_path, packet["vertices"], packet["colors"])

            masked_color_img = np.zeros_like(packet["color_img"])
            masked_color_img[packet["valid_mask_image"]] = packet["color_img"][packet["valid_mask_image"]]

            cv2.imwrite(str(rgb_path), cv2.cvtColor(masked_color_img, cv2.COLOR_RGB2BGR))

            if "native_color_img" in packet:
                native_bgr = cv2.cvtColor(packet["native_color_img"], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(native_rgb_path), native_bgr)

            depth_normalized = packet["depth_img"].astype(np.float32)
            max_depth = depth_normalized.max()
            if max_depth > 0:
                depth_normalized /= max_depth

            depth_to_save = np.clip(depth_normalized * 65535.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(depth_path), depth_to_save)

        if self.frame_times:
            metadata_path = self.current_dir / "frame_capture_times.csv"
            average_ms = sum(self.frame_times) / len(self.frame_times)
            with open(metadata_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["frame_id", "capture_time_ms"])
                for idx, duration in enumerate(self.frame_times, start=1):
                    writer.writerow([idx, round(duration, 2)])
                writer.writerow(["average", round(average_ms, 2)])

        self.buffered_frames.clear()
        self.frames_remaining = 0
        self.is_recording = False
        self.current_dir = None
        self.frame_times.clear()


def _read_pointcloud_ply(ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertex_count = 0
    header_size = 0

    with open(ply_path, "rb") as f:
        while True:
            line = f.readline()
            header_size += len(line)
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("element vertex"):
                vertex_count = int(decoded.split()[-1])
            if decoded == "end_header":
                break

        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("r", "u1"),
                ("g", "u1"),
                ("b", "u1"),
            ]
        )

        f.seek(header_size)
        data = np.fromfile(f, dtype=dtype, count=vertex_count)

    positions = np.vstack((data["x"], data["y"], data["z"])).T.astype(np.float32)
    colors = np.vstack((data["r"], data["g"], data["b"])).T.astype(np.uint8)
    return positions, colors


def compress_pc_set_full(
    set_index: int,
    export_root: Path,
    min_depth: float,
    max_depth: float,
    bandwidth_mbps: float = 40.0,
    target_frame_rate: float = 30.0,
    subsample_frames: bool = True,
    remove_background: bool = True,
) -> Path:
    source_dir = export_root / f"PC_set{set_index}"
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source folder: {source_dir}")

    target_dir = export_root / f"PC_set{set_index}_FULL"
    target_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(source_dir.glob("*.ply"))
    if not ply_files:
        print(f"[FULL] No PLY files found in {source_dir}")
        return target_dir

    stats_rows: List[List[float]] = []

    for ply_path in ply_files:
        print(f"[FULL] Compressing {ply_path.name}")
        vertices, colors = _read_pointcloud_ply(ply_path)
        depth = vertices[:, 2]
        valid_mask = np.isfinite(depth)
        # Toggle background removal
        if remove_background:
            valid_mask &= (depth >= min_depth) & (depth <= max_depth)

        valid_points = int(np.count_nonzero(valid_mask))
        initial_points = valid_points

        effective_ratio = compute_effective_subsample_ratio(
            valid_points, bandwidth_mbps, target_frame_rate
        )
        print(f"[FULL] Effective subsample ratio applied: {effective_ratio:.4f}")

        if subsample_frames:
            valid_mask &= np.random.rand(valid_mask.size) < effective_ratio

        deleted_points = max(0, initial_points - int(np.count_nonzero(valid_mask)))

        vertices = vertices[valid_mask]
        colors = colors[valid_mask]

        _append_stats(stats_rows, ply_path.name, effective_ratio, None, None, deleted_points)

        out_ply = target_dir / ply_path.name
        write_pointcloud_ply(out_ply, vertices, colors)

        rgb_path = ply_path.with_name(f"{ply_path.stem}_rgb.png")
        depth_path = ply_path.with_name(f"{ply_path.stem}_depth.png")

        # Rebuild RGB and depth frames
        mask_flat = None
        frame_shape = None
        mask_active_indices = None

        if rgb_path.exists():
            recorded_rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            recorded_rgb = cv2.cvtColor(recorded_rgb_bgr, cv2.COLOR_BGR2RGB)
            rgb_mask = np.any(recorded_rgb != 0, axis=2)
            mask_flat = rgb_mask.flatten()
            frame_shape = rgb_mask.shape
            mask_active_indices = np.flatnonzero(mask_flat)

        if mask_flat is not None and mask_active_indices is not None:
            if len(mask_active_indices) == valid_mask.size:
                filtered_indices = mask_active_indices[valid_mask]

                rebuilt_rgb = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
                rebuilt_rgb_flat = rebuilt_rgb.reshape(-1, 3)

                rebuilt_rgb_flat[filtered_indices] = colors[: len(filtered_indices)]
                rebuilt_rgb_bgr = cv2.cvtColor(rebuilt_rgb, cv2.COLOR_RGB2BGR)

                rebuilt_depth = np.zeros(frame_shape, dtype=np.float32)
                rebuilt_depth_flat = rebuilt_depth.reshape(-1)
                rebuilt_depth_flat[filtered_indices] = vertices[: len(filtered_indices), 2]

                new_rgb_path = target_dir / rgb_path.name
                new_depth_path = target_dir / depth_path.name

                cv2.imwrite(str(new_rgb_path), rebuilt_rgb_bgr)

                depth_max = rebuilt_depth.max()
                if depth_max > 0:
                    rebuilt_depth /= depth_max
                depth_to_save = np.clip(rebuilt_depth * 65535.0, 0, 65535).astype(np.uint16)
                cv2.imwrite(str(new_depth_path), depth_to_save)
        print(f"[FULL] Exported {out_ply.name}")

    print(f"[FULL] Compression complete. Output at {target_dir}")

    if stats_rows:
        _write_stats_csv(stats_rows, target_dir / "stats.csv", include_codec_metrics=False)

    return target_dir


def _parse_query_point(query: str) -> Tuple[int, int]:
    parts = [p.strip() for p in query.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Query point must look like 'x,y'")
    return int(float(parts[0])), int(float(parts[1]))


def _load_frame_mask(rgb_path: Path, depth_path: Path) -> Tuple[np.ndarray, tuple[int, int], np.ndarray]:
    if rgb_path.exists():
        recorded_rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        recorded_rgb = cv2.cvtColor(recorded_rgb_bgr, cv2.COLOR_BGR2RGB)
        mask = np.any(recorded_rgb != 0, axis=2)
        frame_shape = mask.shape
        mask_flat = mask.flatten()
        active_indices = np.flatnonzero(mask_flat)
        return mask_flat, frame_shape, active_indices, recorded_rgb_bgr, recorded_rgb


def _build_roi_from_query(point_xy: Tuple[int, int], frame_shape: tuple[int, int], box_fraction: float) -> np.ndarray:
    h, w = frame_shape
    half_w = int((w * box_fraction) / 2)
    half_h = int((h * box_fraction) / 2)
    cx, cy = point_xy
    xmin = max(cx - half_w, 0)
    xmax = min(cx + half_w, w)
    ymin = max(cy - half_h, 0)
    ymax = min(cy + half_h, h)
    query_box = NormalizedBoundingBox(
        xmin / w,
        ymin / h,
        xmax / w,
        ymax / h,
    )
    return query_box.to_pixel(img_w=w, img_h=h, as_arr=True)


def _decode(buffer: bytes) -> Tuple[np.ndarray, np.ndarray]:
    if not buffer:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    return dcb.decode_pointcloud(buffer)


def _load_yolo_detector(config: Dict[str, Any]) -> YOLOE: # currently human detector ONLY

    size = config.get("size", "large")
    checkpoint_map = {
        "small": "yoloe-11s-seg.pt",
        "medium": "yoloe-11m-seg.pt",
        "large": "yoloe-11l-seg.pt",
    }
    checkpoint = checkpoint_map.get(size, checkpoint_map["large"])
    detector = YOLOE(checkpoint, verbose=False)

    names = ["human"] # TODO: Maybe extend to more classes later.
    detector.set_classes(names, detector.get_text_pe(names))
    return detector


def _yoloe_pick_class_index(result: Any, roi_point: Optional[Tuple[int, int]]) -> Optional[int]:

    if roi_point is None or result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    if boxes.size == 0:
        return None

    roi_center = np.array(roi_point, dtype=np.float32)
    x_interval = (boxes[:, 0] <= roi_center[0]) & (roi_center[0] <= boxes[:, 2])
    y_interval = (boxes[:, 1] <= roi_center[1]) & (roi_center[1] <= boxes[:, 3])
    roi_isects = x_interval & y_interval

    if not roi_isects.any():
        return None

    classes = result.boxes.cls.cpu().numpy().astype(np.int32)
    return int(classes[np.flatnonzero(roi_isects)[0]])


def _yoloe_segmentation_mask(
    result: Any,
    frame_shape: tuple[int, int],
    roi_point: Optional[Tuple[int, int]],
    roi_init: bool,
    cls_index: Optional[int],
) -> Tuple[np.ndarray, bool, Optional[int]]:

    next_cls_index = cls_index
    next_roi_init = roi_init

    if not next_roi_init:
        selected_class = _yoloe_pick_class_index(result, roi_point)
        if selected_class is not None:
            next_cls_index = selected_class
            next_roi_init = True

    if result.masks is None:
        return np.zeros((*frame_shape, 1), dtype=np.bool_), next_roi_init, next_cls_index

    masks = result.masks.data
    if masks.numel() == 0:
        return np.zeros((*frame_shape, 1), dtype=np.bool_), next_roi_init, next_cls_index

    if next_roi_init and next_cls_index is not None:
        classes = result.boxes.cls.to(dtype=torch.int32)
        class_matches = classes == next_cls_index
        if class_matches.any():
            combined_masks = masks[class_matches].amax(dim=0) > 0.5
        else:
            combined_masks = torch.zeros_like(masks[0], dtype=torch.bool)
    else:
        combined_masks = masks.amax(dim=0) > 0.5

    combined_masks = combined_masks.to(dtype=torch.uint8)[None, None, ...]

    resized_masks = F.interpolate(
        combined_masks,
        size=(frame_shape[0], frame_shape[1]),
        mode="nearest",
    )
    mask = resized_masks.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.bool_)
    return mask, next_roi_init, next_cls_index


# Offline IMPORTANCE compression mode
def compress_pc_set_importance(
    set_indices: List[int],
    export_root: Path,
    min_depth: float,
    max_depth: float,
    query_point: str,
    bandwidth_mbps: float = 40.0,
    target_frame_rate: float = 30.0,
    predictor_type: str = "sam2",
    sam_config: Optional[Dict] = None,
    yolo_config: Optional[Dict[str, Any]] = None,
    remove_background: bool = True,
    in_roi_pos_bits: int = 14,
    out_roi_pos_bits: int = 11,
    in_roi_color_bits: int = 8,
    out_roi_color_bits: int = 6,
) -> List[Path]:

    if predictor_type not in {"sam2", "yolo"}:
        raise ValueError(f"Unknown predictor_type '{predictor_type}'")

    use_sam = predictor_type == "sam2"
    use_yolo = predictor_type == "yolo"

    query_xy = _parse_query_point(query_point) if use_sam else (0, 0)
    device = None
    device_arg = "cuda"
    if use_sam:
        device = sam_config.get("device", "cuda")
    if use_yolo:
        device = yolo_config.get("device", "cuda")

    device_arg = device if isinstance(device, (str, torch.device)) else str(device)
    use_autocast = isinstance(device_arg, torch.device) and device_arg.type == "cuda"

    # choose predictor (sam/yolo)
    with torch.autocast(device_type="cuda", enabled=use_autocast):
        predictor = None
        if use_sam:
            predictor = build_sam2_camera_predictor(
                config_file=sam_config["config_file"],
                config_path=sam_config["config_path"],
                ckpt_path=sam_config["checkpoint_path"],
                device=device_arg,
                image_size=sam_config.get("image_size", 1024),
            )
        if use_yolo:
            predictor = _load_yolo_detector(yolo_config)

        # init encoders
        roi_encoder = DracoWrapper(
            position_quantization_bits=in_roi_pos_bits,
            color_quantization_bits=in_roi_color_bits,
        )
        out_encoder = DracoWrapper(
            position_quantization_bits=out_roi_pos_bits,
            color_quantization_bits=out_roi_color_bits,
        )

        outputs: List[Path] = []

        for set_index in set_indices:
            source_dir = export_root / f"PC_set{set_index}"
            if not source_dir.exists():
                raise FileNotFoundError(f"Missing source folder: {source_dir}")

            target_dir = export_root / f"PC_set{set_index}_IMPORTANCE"
            target_dir.mkdir(parents=True, exist_ok=True)

            ply_files = sorted(source_dir.glob("*.ply"))
            if not ply_files:
                print(f"[IMPORTANCE] No PLY files found in {source_dir}")
                outputs.append(target_dir)
                continue

            current_mask = None
            roi_init = False
            cls_index = None

            # Collect metrics
            stats_rows: List[List[float]] = []

            for ply_path in ply_files:
                print(f"[IMPORTANCE] Compressing {ply_path.name}")

                rgb_path = ply_path.with_name(f"{ply_path.stem}_rgb.png")
                depth_path = ply_path.with_name(f"{ply_path.stem}_depth.png")

                vertices, colors = _read_pointcloud_ply(ply_path)
                depth = vertices[:, 2]
                valid_mask = np.isfinite(depth)

                # Toggle background removal
                if remove_background:
                    valid_mask &= (depth >= min_depth) & (depth <= max_depth)

                mask_flat, frame_shape, active_indices, rgb_frame_bgr, rgb_frame = _load_frame_mask(rgb_path, depth_path)

                # Create roi based on predictor
                if use_sam: # SAM
                    if current_mask is None:
                        roi_pixels = _build_roi_from_query(
                            query_xy, frame_shape, sam_config.get("box_fraction", 0.05)
                        )
                        roi_points = np.array(
                            [
                                [0.5 * (roi_pixels[0] + roi_pixels[2]), 0.5 * (roi_pixels[1] + roi_pixels[3])],
                                [roi_pixels[0], roi_pixels[1]],
                                [roi_pixels[0], roi_pixels[3]],
                                [roi_pixels[2], roi_pixels[1]],
                                [roi_pixels[2], roi_pixels[3]],
                            ],
                            dtype=np.float32,
                        )

                        predictor.load_first_frame(rgb_frame)
                        labels = np.ones(roi_points.shape[0], dtype=np.int32)
                        _, _, out_mask_logits = predictor.add_new_prompt(
                            frame_idx=0,
                            obj_id=(1,),
                            points=roi_points,
                            labels=labels,
                        )
                        current_mask = (
                            (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.bool_)
                        )
                    else:
                        _, out_mask_logits = predictor.track(rgb_frame)
                        current_mask = (
                            (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.bool_)
                        )
                else: # YOLO
                    result = predictor.predict(
                        rgb_frame_bgr,
                        conf=yolo_config.get("conf", 0.25),
                        verbose=False,
                    )[0]
                    current_mask, roi_init, cls_index = _yoloe_segmentation_mask(
                        result,
                        frame_shape,
                        roi_point=None,
                        roi_init=roi_init,
                        cls_index=cls_index,
                    )

                mask_flat_sam = current_mask.reshape(-1)
                if mask_flat_sam.size != mask_flat.size:
                    raise RuntimeError("Mismatch between SAM mask and recorded mask dimensions")

                roi_vertices_mask = mask_flat_sam[active_indices]

                if roi_vertices_mask.size != valid_mask.size:
                    trim_len = min(roi_vertices_mask.size, valid_mask.size)
                    print(
                        f"[IMPORTANCE] Trimming {ply_path.name} masks ({roi_vertices_mask.size})/points ({valid_mask.size}) to {trim_len} to align"
                    )
                    roi_vertices_mask = roi_vertices_mask[:trim_len]
                    valid_mask = valid_mask[:trim_len]
                    vertices = vertices[:trim_len]
                    colors = colors[:trim_len]
                    depth = depth[:trim_len]
                    active_indices = active_indices[:trim_len]

                non_roi_candidates = valid_mask & ~roi_vertices_mask
                subsample_mask = np.zeros_like(valid_mask, dtype=bool)
                non_roi_count = int(np.count_nonzero(non_roi_candidates))

                roi_valid_mask = valid_mask & roi_vertices_mask
                roi_count = int(np.count_nonzero(roi_valid_mask))

                effective_ratio = compute_importance_subsample_ratio(
                    roi_count,
                    non_roi_count,
                    bandwidth_mbps,
                    target_frame_rate,
                    in_roi_pos_bits,
                    in_roi_color_bits,
                    out_roi_pos_bits,
                    out_roi_color_bits,
                )

                print(f"[IMPORTANCE] Effective subsample ratio applied: {effective_ratio:.4f}")

                if non_roi_candidates.any():
                    subsample_mask[non_roi_candidates] = np.random.rand(non_roi_count) < effective_ratio

                out_valid_mask = non_roi_candidates & subsample_mask

                # Log deleted points
                deleted_points = max(0, non_roi_count - int(np.count_nonzero(out_valid_mask)))

                roi_vertices = vertices[roi_valid_mask]
                roi_colors = colors[roi_valid_mask]
                out_vertices = vertices[out_valid_mask]
                out_colors = colors[out_valid_mask]

                # Time encoders
                encode_start = time.perf_counter()
                buffer_roi = roi_encoder.encode(roi_vertices, roi_colors)
                buffer_out = out_encoder.encode(out_vertices, out_colors)
                encode_end = time.perf_counter()

                # Time decoders
                decode_start = time.perf_counter()
                decoded_roi_pos, decoded_roi_col = _decode(buffer_roi)
                decoded_out_pos, decoded_out_col = _decode(buffer_out)
                decode_end = time.perf_counter()

                encode_ms = (encode_end - encode_start) * 1000.0
                decode_ms = (decode_end - decode_start) * 1000.0
                _append_stats(stats_rows, ply_path.name, effective_ratio, encode_ms, decode_ms, deleted_points)

                combined_vertices = np.concatenate([decoded_roi_pos, decoded_out_pos], axis=0)
                combined_colors = np.concatenate([decoded_roi_col, decoded_out_col], axis=0)

                out_ply = target_dir / ply_path.name
                write_pointcloud_ply(out_ply, combined_vertices, combined_colors)

                # Output optional buffers
                # Depth and RGB reconstruction 
                if active_indices.size == valid_mask.size:
                    rebuilt_rgb = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
                    rebuilt_rgb_flat = rebuilt_rgb.reshape(-1, 3)

                    # Map points to original positions
                    roi_pixel_indices = active_indices[roi_valid_mask]
                    out_pixel_indices = active_indices[out_valid_mask]

                    # Fill in reconstructed colors
                    rebuilt_rgb_flat[roi_pixel_indices] = decoded_roi_col[: len(roi_pixel_indices)]
                    rebuilt_rgb_flat[out_pixel_indices] = decoded_out_col[: len(out_pixel_indices)]
                    rebuilt_bgr = cv2.cvtColor(rebuilt_rgb, cv2.COLOR_RGB2BGR)

                    rebuilt_depth = np.zeros(frame_shape, dtype=np.float32)
                    rebuilt_depth_flat = rebuilt_depth.reshape(-1)
                    rebuilt_depth_flat[roi_pixel_indices] = decoded_roi_pos[: len(roi_pixel_indices), 2]
                    rebuilt_depth_flat[out_pixel_indices] = decoded_out_pos[: len(out_pixel_indices), 2]

                    new_rgb_path = target_dir / rgb_path.name
                    new_depth_path = target_dir / depth_path.name

                    cv2.imwrite(str(new_rgb_path), rebuilt_bgr)

                    depth_max = rebuilt_depth.max()
                    if depth_max > 0:
                        rebuilt_depth /= depth_max
                    depth_to_save = np.clip(rebuilt_depth * 65535.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(str(new_depth_path), depth_to_save)
                else:
                    print(
                        f"[IMPORTANCE] Skipping RGB/depth reconstruction for {ply_path.name} because point counts diverged"
                    )

                print(f"[IMPORTANCE] Exported {out_ply.name} with ROI {roi_vertices.shape[0]} / OUT {out_vertices.shape[0]}")

            if stats_rows:
                _write_stats_csv(
                    stats_rows,
                    target_dir / "stats.csv",
                    in_roi_pos_bits=in_roi_pos_bits,
                    out_roi_pos_bits=out_roi_pos_bits,
                    in_roi_color_bits=in_roi_color_bits,
                    out_roi_color_bits=out_roi_color_bits,
                )

            outputs.append(target_dir)

    return outputs
