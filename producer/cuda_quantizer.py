import cupy as cp
import numpy as np
import struct
import os
from enum import Enum

class EncodingMode(Enum):
    HIGH = 0
    MED = 1
    LOW = 2
    
class CudaQuantizer:
    def __init__(self):
        # Kernel for Med/Low Quality (Planar Output)
        self.quantize_kernel = cp.RawKernel(r'''
        typedef unsigned char      uint8_t;
        typedef unsigned short     uint16_t;
        typedef unsigned int       uint32_t;
        typedef int                int32_t;
        typedef unsigned long long uint64_t;
        
        extern "C" __global__
        void quantize_points(
            int num_points,
            const float* __restrict__ vertices,
            const uint8_t* __restrict__ colors,
            int32_t bits_x, int32_t bits_y, int32_t bits_z,
            int32_t bits_r, int32_t bits_g, int32_t bits_b,
            float min_x, float min_y, float min_z,
            float scale_x, float scale_y, float scale_z,
            uint8_t* __restrict__ out_x,
            uint8_t* __restrict__ out_y,
            uint8_t* __restrict__ out_z,
            uint8_t* __restrict__ out_r,
            uint8_t* __restrict__ out_g,
            uint8_t* __restrict__ out_b,
            int stride_x, int stride_y, int stride_z) {

            const int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= num_points) return;
            
            // --- Coordinates ---
            float orig_x = vertices[i * 3 + 0];
            float orig_y = vertices[i * 3 + 1];
            float orig_z = vertices[i * 3 + 2];
            float inv_z = 1.0f / fmaxf(orig_z, 1e-6f);
            
            const float x = __saturatef((orig_x - min_x) * scale_x);
            const float y = __saturatef((orig_y - min_y) * scale_y);
            const float z = __saturatef((inv_z - min_z) * scale_z);
            
            const uint32_t max_x = (1 << bits_x) - 1;
            const uint32_t max_y = (1 << bits_y) - 1;
            const uint32_t max_z = (1 << bits_z) - 1;
            
            const uint32_t qx = (uint32_t)(x * max_x + 0.5f);
            const uint32_t qy = (uint32_t)(y * max_y + 0.5f);
            const uint32_t qz = (uint32_t)(z * max_z + 0.5f);
            
            // Planar writes
            uint32_t offset_x = i * stride_x;
            for (int b = 0; b < stride_x; ++b) out_x[offset_x + b] = (uint8_t)((qx >> (b * 8)) & 0xFF);

            uint32_t offset_y = i * stride_y;
            for (int b = 0; b < stride_y; ++b) out_y[offset_y + b] = (uint8_t)((qy >> (b * 8)) & 0xFF);

            uint32_t offset_z = i * stride_z;
            for (int b = 0; b < stride_z; ++b) out_z[offset_z + b] = (uint8_t)((qz >> (b * 8)) & 0xFF);

            // --- Colors ---
            uint8_t r = colors[i * 3 + 0];
            uint8_t g = colors[i * 3 + 1];
            uint8_t b = colors[i * 3 + 2];
            
            const uint32_t max_r = (1 << bits_r) - 1;
            const uint32_t max_g = (1 << bits_g) - 1;
            const uint32_t max_b = (1 << bits_b) - 1;
            
            uint32_t qr = (uint32_t)((r / float(255)) * max_r + 0.5f);
            uint32_t qg = (uint32_t)((g / float(255)) * max_g + 0.5f);
            uint32_t qb = (uint32_t)((b / float(255)) * max_b + 0.5f);
            
            // MED/LOW color now mirrors HIGH: one byte per channel.
            out_r[i] = (uint8_t)qr;
            out_g[i] = (uint8_t)qg;
            out_b[i] = (uint8_t)qb;
        }
        ''', 'quantize_points')
        
        # Kernel for High Quality (Packed Output)
        self.quantize_hq_kernel = cp.RawKernel(r'''
        typedef unsigned char      uint8_t;
        typedef unsigned int       uint32_t;
        
        extern "C" __global__
        void quantize_hq(
            int num_points,
            const float* __restrict__ vertices,
            const uint8_t* __restrict__ colors,
            float min_x, float min_y, float min_z,
            float scale_x, float scale_y, float scale_z,
            uint32_t* __restrict__ out_coord,
            uint8_t* __restrict__ out_r,
            uint8_t* __restrict__ out_g,
            uint8_t* __restrict__ out_b) {

            const int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= num_points) return;
            
            float orig_x = vertices[i * 3 + 0];
            float orig_y = vertices[i * 3 + 1];
            float orig_z = vertices[i * 3 + 2];
            const float x = __saturatef((orig_x - min_x) * scale_x);
            const float y = __saturatef((orig_y - min_y) * scale_y);
            const float z = __saturatef((orig_z - min_z) * scale_z);
            
            // 11 bits each for X,Y (2047) and 10 for Z (1023)
            const uint32_t max_x = (1 << 11) - 1;
            const uint32_t max_y = (1 << 11) - 1;
            const uint32_t max_z = (1 << 10) - 1;
            
            const uint32_t qx = (uint32_t)(x * max_x + 0.5f);
            const uint32_t qy = (uint32_t)(y * max_y + 0.5f);
            const uint32_t qz = (uint32_t)(z * max_z + 0.5f);
            
            uint32_t packed_coord = 0;
            packed_coord |= qx;
            packed_coord |= (qy << 11);
            packed_coord |= (qz << 22);

            out_coord[i] = packed_coord;

            out_r[i] = colors[i * 3 + 0];
            out_g[i] = colors[i * 3 + 1];
            out_b[i] = colors[i * 3 + 2];
        }
        ''', 'quantize_hq')

    def estimate_buffer_size(self, mode: EncodingMode, num_points: int) -> int:
        # Per-mode byte width is mirrored by the decode path on the client.
        if num_points == 0: return 0 
        
        header_size = 32 
        
        if mode == EncodingMode.HIGH:
            size_coord = num_points * 4

            p1 = num_points
            pad1 = (4 - (p1 % 4)) % 4

            p2 = num_points
            pad2 = (4 - (p2 % 4)) % 4

            p3 = num_points
            
            return header_size + size_coord + (p1 + pad1) + (p2 + pad2) + p3 + ((4 - ((p3 + header_size) % 4)) % 4)

        elif mode == EncodingMode.MED:
            # MED now mirrors HIGH position layout: one packed uint32 per point.
            size_coord = num_points * 4

            p1 = num_points
            pad1 = (4 - (p1 % 4)) % 4

            p2 = num_points
            pad2 = (4 - (p2 % 4)) % 4

            p3 = num_points
            return header_size + size_coord + (p1 + pad1) + (p2 + pad2) + p3 + ((4 - ((p3 + header_size) % 4)) % 4)
        else: # LOW
            sx, sy, sz = 1, 1, 1

        size_x = num_points * sx
        pad_x = (4 - (size_x % 4)) % 4
        
        size_y = num_points * sy
        pad_y = (4 - (size_y % 4)) % 4
        
        size_z = num_points * sz
        pad_z = (4 - (size_z % 4)) % 4
        
        # MED/LOW now store planar RGB like HIGH.
        size_r = num_points
        pad_r = (4 - (size_r % 4)) % 4
        size_g = num_points
        pad_g = (4 - (size_g % 4)) % 4
        size_b = num_points

        total = header_size + (size_x + pad_x) + (size_y + pad_y) + (size_z + pad_z)
        total += (size_r + pad_r) + (size_g + pad_g) + size_b
        final_pad = (4 - (total % 4)) % 4
        
        return total + final_pad
    


    def _compute_low_quant_params(self, points):
        """Compute LOW quantization min/scale using inverse-depth Z."""
        # LOW stores Z in 1/z space, so its range comes from inverse depth.
        d_points = points
        min_vals = cp.amin(d_points, axis=0)
        max_vals = cp.amax(d_points, axis=0)
        inv_z = 1.0 / cp.maximum(d_points[:, 2], cp.float32(1e-6))
        min_v = cp.asnumpy(min_vals)
        max_v = cp.asnumpy(max_vals)
        min_v[2] = float(cp.min(inv_z).get())
        max_v[2] = float(cp.max(inv_z).get())
        diff = (max_v - min_v)
        diff[diff < 1e-6] = 1.0
        scale = 1.0 / diff
        return min_v.astype(np.float32, copy=False), scale.astype(np.float32, copy=False)

    def build_low_dedup_indices(self, points, colors, min_v=None, scale=None):
        """Return sorted first-occurrence indices for LOW pre-quantized XYZ keys."""
        # This mirrors LOW position quantization so dedup aligns with LOW spatial bins.
        if points.shape[0] == 0:
            return cp.empty((0,), dtype=cp.int32)

        d_points = points
        inv_z = 1.0 / cp.maximum(d_points[:, 2], cp.float32(1e-6))

        # Reuse caller-provided params so dedup and encode share the same range.
        if min_v is None or scale is None:
            min_v, scale = self._compute_low_quant_params(d_points)
        else:
            min_v = np.asarray(min_v, dtype=np.float32)
            scale = np.asarray(scale, dtype=np.float32)

        x = cp.clip((d_points[:, 0] - np.float32(min_v[0])) * np.float32(scale[0]), 0.0, 1.0)
        y = cp.clip((d_points[:, 1] - np.float32(min_v[1])) * np.float32(scale[1]), 0.0, 1.0)
        z = cp.clip((inv_z - np.float32(min_v[2])) * np.float32(scale[2]), 0.0, 1.0)

        qx = cp.rint(x * 255.0).astype(cp.uint8)
        qy = cp.rint(y * 255.0).astype(cp.uint8)
        qz = cp.rint(z * 255.0).astype(cp.uint8)
        key = (
            qx.astype(cp.uint64)
            | (qy.astype(cp.uint64) << np.uint64(8))
            | (qz.astype(cp.uint64) << np.uint64(16))
        )

        # unique returns first-hit indices; sorting keeps source order stable.
        _, first_idx = cp.unique(key, return_index=True)
        return cp.sort(first_idx.astype(cp.int32))
    def encode(self, stream, mode: EncodingMode, points, colors, out_pinned=None, min_v=None, scale=None) -> bytes:
        if mode == EncodingMode.HIGH:
            return self.encode_highQ(stream, points, colors, out_pinned)
        elif mode == EncodingMode.MED:
            return self.encode_medQ(stream, points, colors, out_pinned)
        return self.encode_lowQ(stream, points, colors, out_pinned, min_v=min_v, scale=scale)

    def encode_highQ(self, stream, points, colors, out_pinned=None) -> bytes:
        if points.shape[0] == 0: return b""
        num_points = points.shape[0]

        d_points = points
        d_colors = colors

        min_vals = cp.amin(d_points, axis=0)
        max_vals = cp.amax(d_points, axis=0)
        min_v = cp.asnumpy(min_vals)
        max_v = cp.asnumpy(max_vals)
        min_v[2] = float(min_vals[2].get())
        max_v[2] = float(max_vals[2].get())
        diff = (max_v - min_v)
        diff[diff < 1e-6] = 1.0
        scale = 1.0 / diff
        
        buf_coord = cp.empty(num_points, dtype=cp.uint32)
        buf_col_r = cp.empty(num_points, dtype=cp.uint8)
        buf_col_g = cp.empty(num_points, dtype=cp.uint8)
        buf_col_b = cp.empty(num_points, dtype=cp.uint8)

        threads_per_block = 128
        blocks = (num_points + threads_per_block - 1) // threads_per_block
        
        self.quantize_hq_kernel(
            (blocks,), (threads_per_block,),
            (
                num_points, d_points, d_colors,
                np.float32(min_v[0]), np.float32(min_v[1]), np.float32(min_v[2]),
                np.float32(scale[0]), np.float32(scale[1]), np.float32(scale[2]),
                buf_coord, buf_col_r, buf_col_g, buf_col_b
            )
        , stream=stream)
        
        total_size = self.estimate_buffer_size(EncodingMode.HIGH, num_points)
        
        if out_pinned is not None:
            cpu_buffer = out_pinned
        else:
            cpu_buffer = np.zeros(total_size, dtype=np.uint8)
        
        header = struct.pack('<6f', *min_v, *scale)
        header += struct.pack('<2i', EncodingMode.HIGH.value, num_points)
        cpu_buffer[0:32] = np.frombuffer(header, dtype=np.uint8)
        
        current_offset = 32

        end_coord = current_offset + num_points * 4
        if out_pinned is not None and stream is not None:
            buf_coord.get(stream=stream, out=cpu_buffer[current_offset:end_coord].view(np.uint32))
        else:
            cpu_buffer[current_offset:end_coord] = buf_coord.get().view(np.uint8).ravel()
        current_offset = end_coord 

        end_clr_r = current_offset + num_points
        if out_pinned is not None and stream is not None:
            buf_col_r.get(stream=stream, out=cpu_buffer[current_offset:end_clr_r])
        else:
            cpu_buffer[current_offset:end_clr_r] = buf_col_r.get().ravel()
        current_offset = end_clr_r + ((4 - (end_clr_r % 4)) % 4)

        end_clr_g = current_offset + num_points
        if out_pinned is not None and stream is not None:
            buf_col_g.get(stream=stream, out=cpu_buffer[current_offset:end_clr_g])
        else:
            cpu_buffer[current_offset:end_clr_g] = buf_col_g.get().ravel()
        current_offset = end_clr_g + ((4 - (end_clr_g % 4)) % 4)

        end_clr_b = current_offset + num_points
        if out_pinned is not None and stream is not None:
            buf_col_b.get(stream=stream, out=cpu_buffer[current_offset:end_clr_b])
        else:
            cpu_buffer[current_offset:end_clr_b] = buf_col_b.get().ravel()
        
        if out_pinned is not None:
            return cpu_buffer[:total_size]
            
        return cpu_buffer.tobytes()
    
    def encode_medQ(self, stream, points, colors, out_pinned=None) -> bytes:
        # MED now uses the same packed position payload shape as HIGH.
        if points.shape[0] == 0: return b""
        
        num_points = points.shape[0]
        d_points = points
        d_colors = colors

        min_vals = cp.amin(d_points, axis=0)
        max_vals = cp.amax(d_points, axis=0)
        min_v = cp.asnumpy(min_vals)
        max_v = cp.asnumpy(max_vals)
        min_v[2] = float(min_vals[2].get())
        max_v[2] = float(max_vals[2].get())
        diff = (max_v - min_v)
        diff[diff < 1e-6] = 1.0
        scale = 1.0 / diff
        
        buf_coord = cp.empty(num_points, dtype=cp.uint32)
        buf_col_r = cp.empty(num_points, dtype=cp.uint8)
        buf_col_g = cp.empty(num_points, dtype=cp.uint8)
        buf_col_b = cp.empty(num_points, dtype=cp.uint8)

        threads_per_block = 128
        blocks = (num_points + threads_per_block - 1) // threads_per_block
        
        self.quantize_hq_kernel(
            (blocks,), (threads_per_block,),
            (
                num_points, d_points, d_colors,
                np.float32(min_v[0]), np.float32(min_v[1]), np.float32(min_v[2]),
                np.float32(scale[0]), np.float32(scale[1]), np.float32(scale[2]),
                buf_coord, buf_col_r, buf_col_g, buf_col_b
            )
        , stream=stream)
        
        total_size = self.estimate_buffer_size(EncodingMode.MED, num_points)
        
        if out_pinned is not None:
            cpu_buffer = out_pinned
        else:
            cpu_buffer = np.zeros(total_size, dtype=np.uint8)
        
        header = struct.pack('<6f', *min_v, *scale)
        header += struct.pack('<2i', EncodingMode.MED.value, num_points)
        cpu_buffer[0:32] = np.frombuffer(header, dtype=np.uint8)
        
        current_offset = 32

        end_coord = current_offset + (num_points * 4)
        if out_pinned is not None and stream is not None:
            buf_coord.get(stream=stream, out=cpu_buffer[current_offset:end_coord].view(np.uint32))
        else:
            cpu_buffer[current_offset:end_coord] = buf_coord.get().view(np.uint8).ravel()
        current_offset = end_coord

        end_col_r = current_offset + num_points
        if out_pinned is not None and stream is not None:
            buf_col_r.get(stream=stream, out=cpu_buffer[current_offset:end_col_r])
        else:
            cpu_buffer[current_offset:end_col_r] = buf_col_r.get().ravel()
        current_offset = end_col_r + ((4 - (end_col_r % 4)) % 4)

        end_col_g = current_offset + num_points
        if out_pinned is not None and stream is not None:
            buf_col_g.get(stream=stream, out=cpu_buffer[current_offset:end_col_g])
        else:
            cpu_buffer[current_offset:end_col_g] = buf_col_g.get().ravel()
        current_offset = end_col_g + ((4 - (end_col_g % 4)) % 4)

        end_col_b = current_offset + num_points
        if out_pinned is not None and stream is not None:
            buf_col_b.get(stream=stream, out=cpu_buffer[current_offset:end_col_b])
        else:
            cpu_buffer[current_offset:end_col_b] = buf_col_b.get().ravel()
        
        if out_pinned is not None:
            return cpu_buffer[:total_size]
        return cpu_buffer.tobytes()
    
    def encode_lowQ(self, stream, points, colors, out_pinned=None, min_v=None, scale=None) -> bytes:
        # LOW now writes planar RGB to match HIGH decode semantics.
        if points.shape[0] == 0: return b""

        # LOW keeps 8/8/8 position and now uses full 8-bit RGB per channel.
        bits_per_coord = (8, 8, 8)
        bits_per_color = (8, 8, 8)
        sx, sy, sz = 1, 1, 1
        
        num_points = points.shape[0]
        d_points = points
        d_colors = colors

        # Use precomputed LOW params when available to keep dedup and encode aligned.
        if min_v is None or scale is None:
            min_v, scale = self._compute_low_quant_params(d_points)
        else:
            min_v = np.asarray(min_v, dtype=np.float32)
            scale = np.asarray(scale, dtype=np.float32)
        
        buf_x = cp.empty(num_points * sx, dtype=cp.uint8)
        buf_y = cp.empty(num_points * sy, dtype=cp.uint8)
        buf_z = cp.empty(num_points * sz, dtype=cp.uint8)
        buf_col_r = cp.empty(num_points, dtype=cp.uint8)
        buf_col_g = cp.empty(num_points, dtype=cp.uint8)
        buf_col_b = cp.empty(num_points, dtype=cp.uint8)

        threads_per_block = 128
        blocks = (num_points + threads_per_block - 1) // threads_per_block
        
        self.quantize_kernel(
            (blocks,), (threads_per_block,),
            (
                num_points, d_points, d_colors,
                np.int32(bits_per_coord[0]), np.int32(bits_per_coord[1]), np.int32(bits_per_coord[2]),
                np.int32(bits_per_color[0]), np.int32(bits_per_color[1]), np.int32(bits_per_color[2]),
                np.float32(min_v[0]), np.float32(min_v[1]), np.float32(min_v[2]),
                np.float32(scale[0]), np.float32(scale[1]), np.float32(scale[2]),
                buf_x, buf_y, buf_z, buf_col_r, buf_col_g, buf_col_b,
                np.int32(sx), np.int32(sy), np.int32(sz)
            )
        , stream=stream)
        
        total_size = self.estimate_buffer_size(EncodingMode.LOW, num_points)
        
        if out_pinned is not None:
            cpu_buffer = out_pinned
        else:
            cpu_buffer = np.zeros(total_size, dtype=np.uint8)
        
        header = struct.pack('<6f', *min_v, *scale)
        header += struct.pack('<2i', EncodingMode.LOW.value, num_points)
        cpu_buffer[0:32] = np.frombuffer(header, dtype=np.uint8)
        
        current_offset = 32

        end_x = current_offset + (num_points * sx)
        if out_pinned is not None and stream is not None:
            buf_x.get(stream=stream, out=cpu_buffer[current_offset:end_x])
        else:
            cpu_buffer[current_offset:end_x] = buf_x.get().ravel()
        current_offset = end_x + ((4 - (end_x % 4)) % 4)

        end_y = current_offset + (num_points * sy)
        if out_pinned is not None and stream is not None:
            buf_y.get(stream=stream, out=cpu_buffer[current_offset:end_y])
        else:
            cpu_buffer[current_offset:end_y] = buf_y.get().ravel()
        current_offset = end_y + ((4 - (end_y % 4)) % 4)

        end_z = current_offset + (num_points * sz)
        if out_pinned is not None and stream is not None:
            buf_z.get(stream=stream, out=cpu_buffer[current_offset:end_z])
        else:
            cpu_buffer[current_offset:end_z] = buf_z.get().ravel()
        current_offset = end_z + ((4 - (end_z % 4)) % 4)

        end_col_r = current_offset + num_points
        if out_pinned is not None and stream is not None:
            buf_col_r.get(stream=stream, out=cpu_buffer[current_offset:end_col_r])
        else:
            cpu_buffer[current_offset:end_col_r] = buf_col_r.get().ravel()
        current_offset = end_col_r + ((4 - (end_col_r % 4)) % 4)

        end_col_g = current_offset + num_points
        if out_pinned is not None and stream is not None:
            buf_col_g.get(stream=stream, out=cpu_buffer[current_offset:end_col_g])
        else:
            cpu_buffer[current_offset:end_col_g] = buf_col_g.get().ravel()
        current_offset = end_col_g + ((4 - (end_col_g % 4)) % 4)

        end_col_b = current_offset + num_points
        if out_pinned is not None and stream is not None:
            buf_col_b.get(stream=stream, out=cpu_buffer[current_offset:end_col_b])
        else:
            cpu_buffer[current_offset:end_col_b] = buf_col_b.get().ravel()

        if out_pinned is not None:
            return cpu_buffer[:total_size]
        return cpu_buffer.tobytes()
