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
            uint8_t* __restrict__ out_colors,
            int stride_x, int stride_y, int stride_z, int stride_colors) {

            const int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= num_points) return;
            
            // --- Coordinates ---
            float orig_x = vertices[i * 3 + 0];
            float orig_y = vertices[i * 3 + 1];
            float orig_z = vertices[i * 3 + 2];
            
            const float x = __saturatef((orig_x - min_x) * scale_x);
            const float y = __saturatef((orig_y - min_y) * scale_y);
            const float z = __saturatef((orig_z - min_z) * scale_z);
            
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
            
            uint64_t packed_col = 0;
            packed_col |= (uint64_t)qr;
            packed_col |= ((uint64_t)qg << bits_r);
            packed_col |= ((uint64_t)qb << (bits_r + bits_g));

            const uint32_t col_offset = i * stride_colors;
            for (uint32_t byte_i = 0; byte_i < stride_colors; ++byte_i) {
                out_colors[col_offset + byte_i] = (uint8_t)((packed_col >> (byte_i * 8)) & 0xFF);
            }
        }
        ''', 'quantize_points')
        
        # Kernel for High Quality (Planar Output)
        self.quantize_hq_kernel = cp.RawKernel(r'''
        typedef unsigned char      uint8_t;
        typedef unsigned short     uint16_t;
        
        extern "C" __global__
        void quantize_hq(
            int num_points,
            const float* __restrict__ vertices,
            const uint8_t* __restrict__ colors,
            float min_x, float min_y, float min_z,
            float scale_x, float scale_y, float scale_z,
            uint16_t* __restrict__ out_x,
            uint16_t* __restrict__ out_y,
            uint16_t* __restrict__ out_z,
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
            
            const uint16_t max_x = (1 << 13) - 1;
            const uint16_t max_y = (1 << 13) - 1;
            const uint16_t max_z = (1 << 14) - 1;
            
            out_x[i] = (uint16_t)(x * max_x + 0.5f);
            out_y[i] = (uint16_t)(y * max_y + 0.5f);
            out_z[i] = (uint16_t)(z * max_z + 0.5f);

            out_r[i] = colors[i * 3 + 0];
            out_g[i] = colors[i * 3 + 1];
            out_b[i] = colors[i * 3 + 2];
        }
        ''', 'quantize_hq')

    def estimate_buffer_size(self, mode: EncodingMode, num_points: int) -> int:
        if num_points == 0: return 0 
        
        header_size = 32 
        
        if mode == EncodingMode.HIGH:
            sx, sy, sz = 2, 2, 2

            size_x = num_points * sx
            pad_x = (4 - (size_x % 4)) % 4

            size_y = num_points * sy
            pad_y = (4 - (size_y % 4)) % 4

            size_z = num_points * sz
            pad_z = (4 - (size_z % 4)) % 4

            p1 = num_points
            pad1 = (4 - (p1 % 4)) % 4

            p2 = num_points
            pad2 = (4 - (p2 % 4)) % 4

            p3 = num_points
            
            total = header_size + (size_x + pad_x) + (size_y + pad_y) + (size_z + pad_z) + (p1 + pad1) + (p2 + pad2) + p3
            final_pad = (4 - (total % 4)) % 4
            return total + final_pad

        elif mode == EncodingMode.MED:
            sx, sy, sz, scol = 2, 2, 2, 4
        else: # LOW
            sx, sy, sz, scol = 1, 1, 1, 4

        size_x = num_points * sx
        pad_x = (4 - (size_x % 4)) % 4
        
        size_y = num_points * sy
        pad_y = (4 - (size_y % 4)) % 4
        
        size_z = num_points * sz
        pad_z = (4 - (size_z % 4)) % 4
        
        size_col = num_points * scol
        
        total = header_size + (size_x + pad_x) + (size_y + pad_y) + (size_z + pad_z) + size_col
        final_pad = (4 - (total % 4)) % 4
        
        return total + final_pad
    
    def encode(self, stream, mode: EncodingMode, points, colors, out_pinned=None) -> bytes:
        if mode == EncodingMode.HIGH:
            return self.encode_highQ(stream, points, colors, out_pinned)
        elif mode == EncodingMode.MED:
            return self.encode_medQ(stream, points, colors, out_pinned)
        return self.encode_lowQ(stream, points, colors, out_pinned)

    def encode_highQ(self, stream, points, colors, out_pinned=None) -> bytes:
        if points.shape[0] == 0: return b""
        num_points = points.shape[0]

        d_points = points
        d_colors = colors

        min_vals = cp.amin(d_points, axis=0)
        max_vals = cp.amax(d_points, axis=0)
        min_v = cp.asnumpy(min_vals)
        max_v = cp.asnumpy(max_vals)
        diff = (max_v - min_v)
        diff[diff < 1e-6] = 1.0
        scale = 1.0 / diff
        
        buf_x = cp.empty(num_points, dtype=cp.uint16)
        buf_y = cp.empty(num_points, dtype=cp.uint16)
        buf_z = cp.empty(num_points, dtype=cp.uint16)
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
                buf_x, buf_y, buf_z, buf_col_r, buf_col_g, buf_col_b
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

        end_x = current_offset + num_points * 2
        if out_pinned is not None and stream is not None:
            buf_x.get(stream=stream, out=cpu_buffer[current_offset:end_x].view(np.uint16))
        else:
            cpu_buffer[current_offset:end_x] = buf_x.get().view(np.uint8).ravel()
        current_offset = end_x + ((4 - (end_x % 4)) % 4)

        end_y = current_offset + num_points * 2
        if out_pinned is not None and stream is not None:
            buf_y.get(stream=stream, out=cpu_buffer[current_offset:end_y].view(np.uint16))
        else:
            cpu_buffer[current_offset:end_y] = buf_y.get().view(np.uint8).ravel()
        current_offset = end_y + ((4 - (end_y % 4)) % 4)

        end_z = current_offset + num_points * 2
        if out_pinned is not None and stream is not None:
            buf_z.get(stream=stream, out=cpu_buffer[current_offset:end_z].view(np.uint16))
        else:
            cpu_buffer[current_offset:end_z] = buf_z.get().view(np.uint8).ravel()
        current_offset = end_z + ((4 - (end_z % 4)) % 4)

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
        if points.shape[0] == 0: return b""
        
        bits_per_coord = (11, 11, 10)
        bits_per_color = (8, 8, 8)
        sx, sy, sz, scol = 2, 2, 2, 4
        
        num_points = points.shape[0]
        d_points = points
        d_colors = colors

        min_vals = cp.amin(d_points, axis=0)
        max_vals = cp.amax(d_points, axis=0)
        min_v = cp.asnumpy(min_vals)
        max_v = cp.asnumpy(max_vals)
        diff = (max_v - min_v)
        diff[diff < 1e-6] = 1.0
        scale = 1.0 / diff
        
        buf_x = cp.empty(num_points * sx, dtype=cp.uint8)
        buf_y = cp.empty(num_points * sy, dtype=cp.uint8)
        buf_z = cp.empty(num_points * sz, dtype=cp.uint8)
        buf_col = cp.empty(num_points * scol, dtype=cp.uint8)

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
                buf_x, buf_y, buf_z, buf_col,
                np.int32(sx), np.int32(sy), np.int32(sz), np.int32(scol)
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

        end_col = current_offset + (num_points * scol)
        if out_pinned is not None and stream is not None:
            buf_col.get(stream=stream, out=cpu_buffer[current_offset:end_col])
        else:
            cpu_buffer[current_offset:end_col] = buf_col.get().ravel()
        
        if out_pinned is not None:
            return cpu_buffer[:total_size]
        return cpu_buffer.tobytes()
    
    def encode_lowQ(self, stream, points, colors, out_pinned=None) -> bytes:
        if points.shape[0] == 0: return b""
        
        bits_per_coord = (8, 8, 8)
        bits_per_color = (8, 8, 8)
        sx, sy, sz, scol = 1, 1, 1, 4
        
        num_points = points.shape[0]
        d_points = points
        d_colors = colors

        min_vals = cp.amin(d_points, axis=0)
        max_vals = cp.amax(d_points, axis=0)
        min_v = cp.asnumpy(min_vals)
        max_v = cp.asnumpy(max_vals)
        diff = (max_v - min_v)
        diff[diff < 1e-6] = 1.0
        scale = 1.0 / diff
        
        buf_x = cp.empty(num_points * sx, dtype=cp.uint8)
        buf_y = cp.empty(num_points * sy, dtype=cp.uint8)
        buf_z = cp.empty(num_points * sz, dtype=cp.uint8)
        buf_col = cp.empty(num_points * scol, dtype=cp.uint8)

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
                buf_x, buf_y, buf_z, buf_col,
                np.int32(sx), np.int32(sy), np.int32(sz), np.int32(scol)
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

        end_col = current_offset + (num_points * scol)
        if out_pinned is not None and stream is not None:
            buf_col.get(stream=stream, out=cpu_buffer[current_offset:end_col])
        else:
            cpu_buffer[current_offset:end_col] = buf_col.get().ravel()

        if out_pinned is not None:
            return cpu_buffer[:total_size]
        return cpu_buffer.tobytes()
