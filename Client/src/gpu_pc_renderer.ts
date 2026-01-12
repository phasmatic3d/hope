import * as THREE from 'three';

// Data Packet Config
const HEADER_SIZE = 52; 

const pointCloudShader = {
    uniforms: {
        uMin: { value: new THREE.Vector3() },
        uScale: { value: new THREE.Vector3() },
        uBitsPos: { value: new THREE.Vector3(10, 10, 10) }, 
        uBitsCol: { value: new THREE.Vector3(5, 6, 5) },    
    },
    vertexShader: `
        // Attributes (Planar)
        in uint xData;
        in uint yData;
        in uint zData;
        
        in uint packedColor;

        uniform vec3 uMin;
        uniform vec3 uScale;
        uniform vec3 uBitsPos;
        uniform vec3 uBitsCol;

        out vec3 vColor;

        void main() {
            uint bx = uint(uBitsPos.x);
            uint by = uint(uBitsPos.y);
            uint bz = uint(uBitsPos.z);

            uint max_x = (1u << bx) - 1u;
            uint max_y = (1u << by) - 1u;
            uint max_z = (1u << bz) - 1u;

            vec3 norm = vec3(
                float(xData) / float(max_x),
                float(yData) / float(max_y),
                float(zData) / float(max_z)
            );
            
            vec3 pos = (norm / uScale) + uMin;

            uint br = uint(uBitsCol.x);
            uint bg = uint(uBitsCol.y);
            uint bb = uint(uBitsCol.z);

            uint maskR = (1u << br) - 1u;
            uint maskG = (1u << bg) - 1u;
            uint maskB = (1u << bb) - 1u;

            float r = float((packedColor) & maskR) / float(maskR);
            float g = float((packedColor >> br) & maskG) / float(maskG);
            float b = float((packedColor >> (br + bg)) & maskB) / float(maskB);

            vColor = vec3(r, g, b);

            vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
            gl_Position = projectionMatrix * mvPosition;

            float safeDist = max(-mvPosition.z, 0.001);
            gl_PointSize = clamp(3.0 / safeDist, 1.0, 50.0);
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
    const MAX_CHUNKS = 3; 
    const meshPool: THREE.Points[] = [];

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
        mesh.rotateX(Math.PI); 
        scene.add(mesh);
        return mesh;
    }

    for (let i = 0; i < MAX_CHUNKS; i++) {
        meshPool.push(createMesh());
    }

    // Helper to pick correct TypedArray buffer based on bytes-per-component
    function getBufferAttribute(sharedBuf: SharedArrayBuffer, offset: number, count: number, stride: number) {
        if (stride === 4) {
            const arr = new Uint32Array(sharedBuf, offset, count);
            const attr = new THREE.Uint32BufferAttribute(arr, 1);
            attr.gpuType = THREE.IntType; 
            return attr;
        } else if (stride === 2) {
            const arr = new Uint16Array(sharedBuf, offset, count);
            const attr = new THREE.Uint16BufferAttribute(arr, 1);
            attr.gpuType = THREE.IntType;
            return attr;
        } else {
            const arr = new Uint8Array(sharedBuf, offset, count);
            const attr = new THREE.Uint8BufferAttribute(arr, 1);
            attr.gpuType = THREE.IntType;
            return attr;
        }
    }

    function getAxisStride(bits: number) {
        if (bits > 16) return 4;
        if (bits > 8) return 2;
        return 1;
    }

    function updateMeshFromChunk(mesh: THREE.Points, sharedBuf: SharedArrayBuffer, offset: number) {
        const dv = new DataView(sharedBuf, offset, HEADER_SIZE);
        const mat = mesh.material as THREE.ShaderMaterial;
        const geo = mesh.geometry;

        // 1. Read Header
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

        mat.uniforms.uMin.value.set(minX, minY, minZ);
        mat.uniforms.uScale.value.set(scaleX, scaleY, scaleZ);
        mat.uniforms.uBitsPos.value.set(bx, by, bz);
        mat.uniforms.uBitsCol.value.set(br, bg, bb);

        // 2. Calculate Strides and Offsets
        const sx = getAxisStride(bx);
        const sy = getAxisStride(by);
        const sz = getAxisStride(bz);
        
        // Color stride logic matches Python
        const rawColBytes = Math.ceil((br + bg + bb) / 8);
        const scol = (rawColBytes === 3) ? 4 : rawColBytes;

        // Offsets
        let currentOffset = offset + HEADER_SIZE;

        // X Plane
        geo.setAttribute('xData', getBufferAttribute(sharedBuf, currentOffset, numPoints, sx));
        currentOffset += (sx * numPoints);
        currentOffset += (4 - (currentOffset % 4)) % 4; // Skip padding

        // Y Plane
        geo.setAttribute('yData', getBufferAttribute(sharedBuf, currentOffset, numPoints, sy));
        currentOffset += (sy * numPoints);
        currentOffset += (4 - (currentOffset % 4)) % 4; // Skip padding

        // Z Plane
        geo.setAttribute('zData', getBufferAttribute(sharedBuf, currentOffset, numPoints, sz));
        currentOffset += (sz * numPoints);
        currentOffset += (4 - (currentOffset % 4)) % 4; // Skip padding

        // Color Plane
        geo.setAttribute('packedColor', getBufferAttribute(sharedBuf, currentOffset, numPoints, scol));

        // 3. Draw
        geo.setDrawRange(0, numPoints);
        // Mark all as needing update
        geo.attributes.xData.needsUpdate = true;
        geo.attributes.yData.needsUpdate = true;
        geo.attributes.zData.needsUpdate = true;
        geo.attributes.packedColor.needsUpdate = true;
    }

    let pendingFrameId = -1;
    let accumulatedChunks: {offset: number, length: number}[] = [];

    return (
        sharedBuf: SharedArrayBuffer,
        chunk: { offset: number; length: number },
        numChunks: number,
        frameId: number
    ) => {
        const tStart = performance.now();

        if (frameId !== pendingFrameId) {
            pendingFrameId = frameId;
            accumulatedChunks = [];
        }

        accumulatedChunks.push(chunk);

        if (accumulatedChunks.length === numChunks) {
            for (let m of meshPool) m.visible = false;

            for (let i = 0; i < accumulatedChunks.length; i++) {
                if (i >= MAX_CHUNKS) break;
                
                const ch = accumulatedChunks[i];
                const mesh = meshPool[i];
                
                updateMeshFromChunk(mesh, sharedBuf, ch.offset);
                mesh.visible = true;
            }
            accumulatedChunks = [];
        }
        return performance.now() - tStart;
    }
}