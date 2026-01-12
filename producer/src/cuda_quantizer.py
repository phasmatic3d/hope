import cupy as cp
import numpy as np
import struct
import os

class CudaQuantizer:
    def __init__(self):
        # Updated Kernel: Writes to separate X, Y, Z, Color buffers
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
            
            // --- 1. Quantize Coordinates ---
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
            
            // --- 2. Write X Plane ---
            uint32_t offset_x = i * stride_x;
            for (int b = 0; b < stride_x; ++b) {
                out_x[offset_x + b] = (uint8_t)((qx >> (b * 8)) & 0xFF);
            }

            // --- 3. Write Y Plane ---
            uint32_t offset_y = i * stride_y;
            for (int b = 0; b < stride_y; ++b) {
                out_y[offset_y + b] = (uint8_t)((qy >> (b * 8)) & 0xFF);
            }

            // --- 4. Write Z Plane ---
            uint32_t offset_z = i * stride_z;
            for (int b = 0; b < stride_z; ++b) {
                out_z[offset_z + b] = (uint8_t)((qz >> (b * 8)) & 0xFF);
            }

            // --- 5. Write Color Plane (Unchanged Logic) ---
            uint8_t r = colors[i * 3 + 0];
            uint8_t g = colors[i * 3 + 1];
            uint8_t b = colors[i * 3 + 2];
            
            uint32_t qr = r >> (8 - bits_r);
            uint32_t qg = g >> (8 - bits_g);
            uint32_t qb = b >> (8 - bits_b);
            
            uint64_t packed_col = 0;
            packed_col |= (uint64_t)qr;
            packed_col |= ((uint64_t)qg << bits_r);
            packed_col |= ((uint64_t)qb << (bits_r + bits_g));

            const uint32_t col_offset = i * stride_colors;
            for (uint32_t b = 0; b < stride_colors; ++b) {
                out_colors[col_offset + b] = (uint8_t)((packed_col >> (b * 8)) & 0xFF);
            }
        }
        ''', 'quantize_points')
        
        self.warmup()
        self.dump_b = False

    def warmup(self):
        print("Warming up GPU kernel...")
        dummy_points = np.random.rand(10, 3)
        dummy_colors = (np.random.rand(10, 3) * 255).astype(np.uint8)
        self.encode(dummy_points, dummy_colors, (10, 10, 10), (8, 8, 8))
        cp.cuda.Stream.null.synchronize()
        print("GPU Warmup Complete.")
    
    # Helper to determine bytes needed (1, 2, or 4) based on bit depth
    def get_axis_stride(self, bits):
        if bits > 16: return 4
        if bits > 8:  return 2
        return 1

    def get_strides(self, bits_per_coord, bits_per_color):
        sx = self.get_axis_stride(bits_per_coord[0])
        sy = self.get_axis_stride(bits_per_coord[1])
        sz = self.get_axis_stride(bits_per_coord[2])
        
        # Color logic remains: pad 3-byte colors to 4 bytes for JS compatibility
        raw_col_bytes = (sum(bits_per_color) + 7) // 8
        scol = 4 if raw_col_bytes == 3 else raw_col_bytes
        
        return sx, sy, sz, scol
    
    def estimate_buffer_size(self, points: np.ndarray, colors: np.ndarray, bits_per_coord : tuple, bits_per_color: tuple) -> int:
        if points.shape[0] == 0: return 0 
        
        sx, sy, sz, scol = self.get_strides(bits_per_coord, bits_per_color)
        num_points = points.shape[0]
        header_size = 52
        
        # Size + Padding for each plane
        size_x = sx * num_points
        pad_x  = (4 - (size_x % 4)) % 4

        size_y = sy * num_points
        pad_y  = (4 - (size_y % 4)) % 4

        size_z = sz * num_points
        pad_z  = (4 - (size_z % 4)) % 4

        size_col = scol * num_points
        
        total = header_size + (size_x + pad_x) + (size_y + pad_y) + (size_z + pad_z) + size_col
        
        # Final padding for next chunk alignment
        final_pad = (4 - (total % 4)) % 4
        return total + final_pad
    
    def encode(self, points: np.ndarray, colors: np.ndarray, bits_per_coord : tuple, bits_per_color: tuple) -> bytes:
        if points.shape[0] == 0: return b""
        if sum(bits_per_coord) > 96: return b"" # 32 bits per axis max
        if any(c > 8 for c in bits_per_color) or sum(bits_per_color) > 32: return b""
                
        num_points = points.shape[0]

        d_points = cp.asarray(points, dtype=cp.float32)
        d_colors = cp.asarray(colors, dtype=cp.uint8)

        min_vals = cp.amin(d_points, axis=0)
        max_vals = cp.amax(d_points, axis=0)
        
        min_v = cp.asnumpy(min_vals)
        max_v = cp.asnumpy(max_vals)
        
        diff = (max_v - min_v)
        diff[diff < 1e-6] = 1.0
        scale = 1.0 / diff
        
        sx, sy, sz, scol = self.get_strides(bits_per_coord, bits_per_color)
        
        # Alloc separate buffers
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
        )
        
        # 2. Allocate ONE CPU buffer (Zero-Copy Preparation)
        # We calculate the exact size required including all headers and padding
        total_size = self.estimate_buffer_size(points, colors, bits_per_coord, bits_per_color)
        
        # Use a bytearray or numpy array on CPU (pinned memory would be even faster, but this is good enough)
        cpu_buffer = np.zeros(total_size, dtype=np.uint8)
        
        # 3. Write Header
        header = struct.pack('<6f', *min_v, *scale)
        header += struct.pack('<6i', *bits_per_coord, *bits_per_color)
        header += struct.pack('<1i', num_points)
        cpu_buffer[0:52] = np.frombuffer(header, dtype=np.uint8)
        
        current_offset = 52

        # 4. Transfer Planes directly to their slots
        # Note: We use slice assignment. This performs the download from GPU directly into the correct place.
        # X Plane
        end_x = current_offset + (num_points * sx)
        cpu_buffer[current_offset:end_x] = buf_x.get().ravel() # Transfer 1
        current_offset = end_x + ((4 - (end_x % 4)) % 4) # Skip padding

        # Y Plane
        end_y = current_offset + (num_points * sy)
        cpu_buffer[current_offset:end_y] = buf_y.get().ravel() # Transfer 2
        current_offset = end_y + ((4 - (end_y % 4)) % 4)

        # Z Plane
        end_z = current_offset + (num_points * sz)
        cpu_buffer[current_offset:end_z] = buf_z.get().ravel() # Transfer 3
        current_offset = end_z + ((4 - (end_z % 4)) % 4)

        # Color Plane
        end_col = current_offset + (num_points * scol)
        cpu_buffer[current_offset:end_col] = buf_col.get().ravel() # Transfer 4
        
        # Return the buffer as bytes
        return cpu_buffer.tobytes()