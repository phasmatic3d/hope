import type { DecoderMessage } from './types';

let sharedEncodedView: Uint8Array;
let decodedPosView: Float32Array;
let decodedColView: Uint8Array;

self.onmessage = async (ev: MessageEvent<any>) => {
    const data = ev.data;

    // 1. Initialization
    if (data.type === 'init') {
        sharedEncodedView = new Uint8Array(data.sharedEncodedBuffer);
        decodedPosView    = new Float32Array(data.decodedPosBuffer);
        decodedColView    = new Uint8Array(data.decodedColBuffer);
        console.log("[Worker] Init complete");
        return;
    }

    // 2. Decode Request
    if (data.type === 'decode') {
        const { chunks, frameId } = data;
        let totalPointsDecoded = 0;
        const startTime = performance.now();

        for (const chunk of chunks) {
            const { offset, length } = chunk;

            if (offset + length > sharedEncodedView.byteLength) {
                console.error(`[Worker] OOB Read! Offset ${offset} + Len ${length} > Buffer ${sharedEncodedView.byteLength}`);
                continue;
            }

            const chunkView = new DataView(
                sharedEncodedView.buffer, 
                sharedEncodedView.byteOffset + offset, 
                length
            );

            // --- A. Header (40 Bytes) ---
            const minX = chunkView.getFloat32(0, true);
            const minY = chunkView.getFloat32(4, true);
            const minZ = chunkView.getFloat32(8, true);

            const scaleX = chunkView.getFloat32(12, true);
            const scaleY = chunkView.getFloat32(16, true);
            const scaleZ = chunkView.getFloat32(20, true);

            const bitsX = chunkView.getInt32(24, true);
            const bitsY = chunkView.getInt32(28, true);
            const bitsZ = chunkView.getInt32(32, true);

            const num_points = chunkView.getInt32(36, true);

            if (num_points <= 0 || num_points > 1000000) {
                if (frameId % 60 === 0) console.warn("[Worker] Garbage header detected. Skipping chunk.");
                continue;
            }

            // --- B. Setup Offsets ---
            const HEADER_SIZE = 40;
            const strideCoord = (bitsX + bitsY + bitsZ + 7) >> 3;
            
            // FIX 1: YCoCg packs into 16 bits (2 bytes)
            const strideColor = 2; 

            const maxX = (1 << bitsX) - 1;
            const maxY = (1 << bitsY) - 1;
            const maxZ = (1 << bitsZ) - 1;
            const maskX = BigInt(maxX);
            const maskY = BigInt(maxY);
            const maskZ = BigInt(maxZ);
            
            let coordOffset = HEADER_SIZE;
            let colorOffset = HEADER_SIZE + (num_points * strideCoord);

            // --- C. Decode Loop ---
            for (let i = 0; i < num_points; i++) {
                const writeIndex = totalPointsDecoded + i;

                // 1. Unpack Coordinates
                let packedCoord = 0n;
                for (let b = 0; b < strideCoord; b++) {
                    const byte = chunkView.getUint8(coordOffset + b);
                    packedCoord |= (BigInt(byte) << BigInt(b * 8));
                }
                coordOffset += strideCoord;

                const qx = Number(packedCoord & maskX);
                const qy = Number((packedCoord >> BigInt(bitsX)) & maskY);
                const qz = Number((packedCoord >> BigInt(bitsX + bitsY)) & maskZ);

                decodedPosView[writeIndex * 3 + 0] = (qx / maxX) / scaleX + minX;
                decodedPosView[writeIndex * 3 + 1] = (qy / maxY) / scaleY + minY;
                // Z was encoded in inverse-depth space, then converted back here.
                const invZ = (qz / maxZ) / scaleZ + minZ;
                decodedPosView[writeIndex * 3 + 2] = 1.0 / Math.max(invZ, 1e-6);

                // 2. Unpack Colors (YCoCg)
                // Read 2 bytes (16 bits)
                const packed = chunkView.getUint16(colorOffset, true);
                
                if (false) {
                    // Unpack: Y (8 bits) | Co (4 bits) | Cg (4 bits)
                    const y  = (packed >> 8) & 0xFF;
                    const co = (((packed >> 4) & 0x0F) << 4) - 128; // Expand 4 bits to ~8 bits
                    const cg = (((packed >> 0) & 0x0F) << 4) - 128;

                    // Inverse YCoCg
                    const t = y - (cg >> 1);
                    const g = y + (cg >> 1);
                    const b = t - (co >> 1);
                    const r = b + co;

                    decodedColView[writeIndex * 3 + 0] = Math.max(0, Math.min(255, r));
                    decodedColView[writeIndex * 3 + 1] = Math.max(0, Math.min(255, g));
                    decodedColView[writeIndex * 3 + 2] = Math.max(0, Math.min(255, b));
                } else {
                    const r5 = (packed >> 11) & 0x1F;
                    const g6 = (packed >> 5)  & 0x3F;
                    const b5 = (packed >> 0)  & 0x1F;

                    // Scale up to 8-bit (0-255)
                    // (val << 3) | (val >> 2) replicates the top bits to fill the bottom bits
                    // e.g. 11111 becomes 11111111 (Full White) instead of 11111000
                    const r8 = (r5 << 3) | (r5 >> 2);
                    const g8 = (g6 << 2) | (g6 >> 4);
                    const b8 = (b5 << 3) | (b5 >> 2);

                    decodedColView[writeIndex * 3 + 0] = r8;
                    decodedColView[writeIndex * 3 + 1] = g8;
                    decodedColView[writeIndex * 3 + 2] = b8;
                }
                
                colorOffset += strideColor; 
            }

            totalPointsDecoded += num_points;
        }

        const endTime = performance.now();
        const decodeTimeMs = endTime - startTime;

        self.postMessage({ type: 'decoded', numPoints: totalPointsDecoded, frameId, decodeTimeMs } as DecoderMessage);
    }
};