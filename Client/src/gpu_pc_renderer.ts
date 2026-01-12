import * as THREE from 'three';

// Data Packet Config
const HEADER_SIZE = 52; // 6 floats (24) + 6 ints (24) + 1 int (4)

// Shader for decompressing points on the GPU
const pointCloudShader = {
    uniforms: {
        uMin: { value: new THREE.Vector3() },
        uScale: { value: new THREE.Vector3() },
        uBitsPos: { value: new THREE.Vector3(10, 10, 10) }, // x, y, z bits
        uBitsCol: { value: new THREE.Vector3(5, 6, 5) },    // r, g, b bits
        uPointSize: { value: 1.0 },
    },
    vertexShader: `
        in uint packedPos;
        in uint packedColor;

        uniform vec3 uMin;
        uniform vec3 uScale;
        uniform vec3 uBitsPos;
        uniform vec3 uBitsCol;
        uniform float uPointSize;

        out vec3 vColor;

        void main() {
            uint bx = uint(uBitsPos.x);
            uint by = uint(uBitsPos.y);
            uint bz = uint(uBitsPos.z);

            // Create masks: (1 << n) - 1
            uint maskX = (1u << bx) - 1u;
            uint maskY = (1u << by) - 1u;
            uint maskZ = (1u << bz) - 1u;

            // Extract integer coords
            uint qx = packedPos & maskX;
            uint qy = (packedPos >> bx) & maskY;
            uint qz = (packedPos >> (bx + by)) & maskZ;

            // Normalize to 0.0 - 1.0
            vec3 norm = vec3(float(qx)/float(maskX), float(qy)/float(maskY), float(qz)/float(maskZ));

            // De-quantize: pos = norm / scale + min
            vec3 pos = (norm / uScale) + uMin;

            // --- 2. Dynamic Bit Unpacking (Color) ---
            uint br = uint(uBitsCol.x);
            uint bg = uint(uBitsCol.y);
            uint bb = uint(uBitsCol.z);

            uint maskR = (1u << br) - 1u;
            uint maskG = (1u << bg) - 1u;
            uint maskB = (1u << bb) - 1u;

            // Note: Colors are often packed R | G | B or B | G | R depending on endianness.
            // Assuming standard little-endian packing order (R lowest bits)
            float r = float((packedColor) & maskR) / float(maskR);
            float g = float((packedColor >> br) & maskG) / float(maskG);
            float b = float((packedColor >> (br + bg)) & maskB) / float(maskB);

            vColor = vec3(r, g, b);

            gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
            gl_PointSize = uPointSize;
        }
    `,
    fragmentShader: `
        precision highp float;
        in vec3 vColor;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(vColor, 1.0);
        }
    `
};

export function createGPUPointCloud(scene: THREE.Scene) {
    const MAX_CHUNKS = 5; // Support ROI + Mid + Out + potential fragmentation
    const meshPool: THREE.Points[] = [];

    // Factory to create meshes
    function createMesh() {
        const geo = new THREE.BufferGeometry();
        
        const material = new THREE.ShaderMaterial({
            uniforms: THREE.UniformsUtils.clone(pointCloudShader.uniforms),
            vertexShader: pointCloudShader.vertexShader,
            fragmentShader: pointCloudShader.fragmentShader,
            glslVersion: THREE.GLSL3
        });

        const mesh = new THREE.Points(geo, material);
        mesh.frustumCulled = false;
        mesh.visible = false;
        // Adjust for RealSense coordinate system if needed (often inverted Y or Z)
        mesh.rotateX(Math.PI); 
        scene.add(mesh);
        return mesh;
    }

    // Init Pool
    for (let i = 0; i < MAX_CHUNKS; i++) {
        meshPool.push(createMesh());
    }

    let currentFrameId = -1;
    let chunkCursor = 0;

    return (
        sharedBuf: SharedArrayBuffer,
        chunk: { offset: number; length: number },
        numChunks: number,
        frameId: number
    ) => {
        const tStart = performance.now();

        // New frame logic
        if (frameId !== currentFrameId) {
            for (let m of meshPool) m.visible = false;
            currentFrameId = frameId;
            chunkCursor = 0;
        }

        if (chunkCursor >= MAX_CHUNKS) return;

        const mesh = meshPool[chunkCursor];
        const geo = mesh.geometry;
        const mat = mesh.material as THREE.ShaderMaterial;

        // 1. Read Header (52 Bytes)
        const dv = new DataView(sharedBuf, chunk.offset, HEADER_SIZE);

        const minX = dv.getFloat32(0, true);
        const minY = dv.getFloat32(4, true);
        const minZ = dv.getFloat32(8, true);
        const scaleX = dv.getFloat32(12, true);
        const scaleY = dv.getFloat32(16, true);
        const scaleZ = dv.getFloat32(20, true);

        const bx = dv.getInt32(24, true);
        const by = dv.getInt32(28, true);
        const bz = dv.getInt32(32, true);
        
        const br = dv.getInt32(36, true);
        const bg = dv.getInt32(40, true);
        const bb = dv.getInt32(44, true);

        const numPoints = dv.getInt32(48, true);

        // 2. Identify Material Layer (A, B, C)
        const totalBitsPos = bx + by + bz;
        const totalBitsCol = br + bg + bb;

        // --- Material A: High Quality (ROI) ---
        if (totalBitsPos > 16) {
             // Expecting 32-bit coordinates (Uint32)
             // High visual fidelity
             mat.uniforms.uPointSize.value = 4.0;
        } 
        // --- Material B: Mid Quality (Silhouette) ---
        else if (totalBitsPos > 8) {
             // Expecting 16-bit coordinates (Uint16)
             mat.uniforms.uPointSize.value = 3.0;
        }
        // --- Material C: Low Quality (Background) ---
        else {
             // Expecting 8-bit coordinates (Uint8)
             mat.uniforms.uPointSize.value = 2.0;
        }

        // 3. Update Uniforms
        mat.uniforms.uMin.value.set(minX, minY, minZ);
        mat.uniforms.uScale.value.set(scaleX, scaleY, scaleZ);
        mat.uniforms.uBitsPos.value.set(bx, by, bz);
        mat.uniforms.uBitsCol.value.set(br, bg, bb);

        // 4. Bind Attributes Dynamically
        const stridePos = Math.ceil(totalBitsPos / 8);
        const strideCol = Math.ceil(totalBitsCol / 8);
        
        // Position Attribute
        const posByteOffset = chunk.offset + HEADER_SIZE;
        let posAttr: THREE.BufferAttribute;

        if (stridePos > 2) {
            // 32-bit Int
            const buf = new Uint32Array(sharedBuf, posByteOffset, numPoints);
            posAttr = new THREE.Uint32BufferAttribute(buf, 1);
        } else if (stridePos > 1) {
            // 16-bit Int
            const buf = new Uint16Array(sharedBuf, posByteOffset, numPoints);
            posAttr = new THREE.Uint16BufferAttribute(buf, 1);
        } else {
            // 8-bit Int
            const buf = new Uint8Array(sharedBuf, posByteOffset, numPoints);
            posAttr = new THREE.Uint8BufferAttribute(buf, 1);
        }
        // Critical for shader to read 'uint' not 'float'
        posAttr.gpuType = THREE.IntType; 
        geo.setAttribute('packedPos', posAttr);

        const rawPosSize = numPoints * stridePos;
        // The server adds padding to ensure the Position block ends on a 4-byte boundary
        const paddedPosSize = (rawPosSize + 3) & ~3; 
        
        const colByteOffset = posByteOffset + paddedPosSize;
        let colAttr: THREE.BufferAttribute;
        const serverStrideCol = (strideCol > 2) ? 4 : strideCol;

        if (serverStrideCol > 2) {
             const buf = new Uint32Array(sharedBuf, colByteOffset, numPoints);
             colAttr = new THREE.Uint32BufferAttribute(buf, 1);
        } else if (serverStrideCol > 1) {
             const buf = new Uint16Array(sharedBuf, colByteOffset, numPoints);
             colAttr = new THREE.Uint16BufferAttribute(buf, 1);
        } else {
             const buf = new Uint8Array(sharedBuf, colByteOffset, numPoints);
             colAttr = new THREE.Uint8BufferAttribute(buf, 1);
        }
        
        colAttr.gpuType = THREE.IntType;
        geo.setAttribute('packedColor', colAttr);

        // 5. Draw
        geo.setDrawRange(0, numPoints);
        geo.attributes.packedPos.needsUpdate = true;
        geo.attributes.packedColor.needsUpdate = true;
        
        mesh.visible = true;
        chunkCursor++;
        return performance.now() - tStart;
    }
}