import cupy as cp
import numpy as np
import struct

class CudaQuantizer:
    def __init__(self):
        self.quantize_kernel = cp.RawKernel(r'''
        static __forceinline__ __device__ float clamp(float x, float a, float b) {
            return fminf(fmaxf(x, a), b);
        }
        
        extern "C" __global__
        void quantize_points(
            int num_points,
            const float* __restrict__ vertices,
            const unsigned char* __restrict__ colors,
            unsigned char* __restrict__ output,
            float min_x, float min_y, float min_z, float scale) {

            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= num_points) return;
            
            float x = clamp((vertices[i * 3 + 0] - min_x) * scale, 0.f, 1.f);
            float y = clamp((vertices[i * 3 + 1] - min_y) * scale, 0.f, 1.f);
            float z = clamp((vertices[i * 3 + 2] - min_z) * scale, 0.f, 1.f);
#if 0
            unsigned short qx = (unsigned short)(x);
            unsigned short qy = (unsigned short)(y);
            unsigned short qz = (unsigned short)(z);

            int offset = i * 9;
            
            // Position
            reinterpret_cast<unsigned short*>(&output[offset + 0])[0] = qx;
            reinterpret_cast<unsigned short*>(&output[offset + 2])[0] = qy;
            reinterpret_cast<unsigned short*>(&output[offset + 4])[0] = qz;

            // Color
            output[offset + 6] = colors[i * 3 + 0];
            output[offset + 7] = colors[i * 3 + 1];
            output[offset + 8] = colors[i * 3 + 2];
#endif
        }
        ''', 'quantize_points')

    def encode(self, points: np.ndarray, colors: np.ndarray) -> bytes:
        """
        Input: 
            points (CPU numpy N,3 float32)
            colors (CPU numpy N,3 uint8)
        Output:
            bytes (Header + Packed Data)
        """
        if points.shape[0] == 0:
            return b""

        num_points = points.shape[0]

        d_points = cp.asarray(points)
        d_colors = cp.asarray(colors)

        min_vals = cp.amin(d_points, axis=0)
        max_vals = cp.amax(d_points, axis=0)
        
        min_v = cp.asnumpy(min_vals)
        max_v = cp.asnumpy(max_vals)
        
        diff = max_v - min_v
        max_range = np.max(diff)

        # 3. Allocate Output on GPU (N * 9 bytes)
        d_output = cp.empty(num_points * 9, dtype=cp.uint8)

        # 4. Run Kernel
        threads_per_block = 256
        blocks = (num_points + threads_per_block - 1) // threads_per_block
        
        self.quantize_kernel(
            (blocks,), (threads_per_block,),
            (
                num_points,
                d_points, d_colors, d_output,
                float(min_v[0]), float(min_v[1]), float(min_v[2]),
                float(max_range)
            )
        )

        packed_data = d_output.get().tobytes()

        # 6. Create Header (16 Bytes)
        # [MinX (4), MinY (4), MinZ (4), MaxRange (4)]
        # We need these to decode on client side
        header = struct.pack('<4f', min_v[0], min_v[1], min_v[2], max_range)

        return header + packed_data