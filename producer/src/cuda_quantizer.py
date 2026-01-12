import cupy as cp
import numpy as np
import struct
import os

class CudaQuantizer:
    def __init__(self):
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
            uint8_t* __restrict__ quantized_coords,
            uint8_t* __restrict__ quantized_colors,
            int stride_colors) {

            const int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= num_points) return;
            
            // --- 1. Coordinates ---
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
            
            uint64_t packed = 0;
            packed |= (uint64_t)qx;
            packed |= ((uint64_t)qy << bits_x);
            packed |= ((uint64_t)qz << (bits_x + bits_y));
             
            const uint32_t total_bytes = (bits_x + bits_y + bits_z + 7) >> 3;
            const uint32_t byte_offset = i * total_bytes;
            
            for (uint32_t b = 0; b < total_bytes; ++b) {
                quantized_coords[byte_offset + b] = (uint8_t)((packed >> (b * 8)) & 0xFF);
            }

            // --- 2. Colors ---
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
                quantized_colors[col_offset + b] = (uint8_t)((packed_col >> (b * 8)) & 0xFF);
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
    
    def get_strides(self, bits_per_coord, bits_per_color):
        stride_coords = (sum(bits_per_coord) + 7) // 8
        raw_col_bytes = (sum(bits_per_color) + 7) // 8
        stride_colors = 4 if raw_col_bytes == 3 else raw_col_bytes
        return stride_coords, stride_colors
    
    def estimate_buffer_size(self, points: np.ndarray, colors: np.ndarray, bits_per_coord : tuple, bits_per_color: tuple) -> int:
        if points.shape[0] == 0: return 0 
        
        stride_coords, stride_colors = self.get_strides(bits_per_coord, bits_per_color)
        num_points = points.shape[0]
        header_size = 52
        
        coords_size = stride_coords * num_points
        coords_padding = (4 - (coords_size % 4)) % 4
        
        colors_size = stride_colors * num_points
        
        total_payload = header_size + coords_size + coords_padding + colors_size
        
        final_padding = (4 - (total_payload % 4)) % 4
        
        return total_payload + final_padding
    
    def encode(self, points: np.ndarray, colors: np.ndarray, bits_per_coord : tuple, bits_per_color: tuple) -> bytes:
        if points.shape[0] == 0: return b""
        if sum(bits_per_coord) > 32: return b""
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
        
        stride_coords, stride_colors = self.get_strides(bits_per_coord, bits_per_color)
        
        quantized_coords = cp.empty(num_points * stride_coords, dtype=cp.uint8)
        quantized_colors = cp.empty(num_points * stride_colors, dtype=cp.uint8)

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
                quantized_coords, quantized_colors,
                np.int32(stride_colors)
            )
        )
        
        header = struct.pack('<6f', *min_v, *scale)
        header += struct.pack('<6i', *bits_per_coord, *bits_per_color)
        header += struct.pack('<1i', num_points)
        
        coords_bytes = quantized_coords.get().tobytes()
        colors_bytes = quantized_colors.get().tobytes()
        
        coords_padding_len = (4 - (len(coords_bytes) % 4)) % 4
        coords_padding = b'\0' * coords_padding_len
        
        payload = header + coords_bytes + coords_padding + colors_bytes

        final_padding_len = (4 - (len(payload) % 4)) % 4
        if final_padding_len > 0:
            payload += b'\0' * final_padding_len
            
        return payload