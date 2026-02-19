import * as THREE from 'three';

const HEADER_SIZE = 32;

const MODE_HIGH = 0;
const MODE_MED = 1;
const MODE_LOW = 2;

const highQShader = {
    uniforms: {
        uMin: { value: new THREE.Vector3() },
        uScale: { value: new THREE.Vector3() }
    },
    vertexShader: `
        precision highp int;
        precision highp float;

        in uint packedPos; // 32-bit: Z(10) | Y(11) | X(11)
        in uint rData;
        in uint gData;
        in uint bData;

        uniform vec3 uMin;
        uniform vec3 uScale;

        out vec3 vColor;

        void main() {
            float x = float(packedPos & 2047u) / 2047.0;
            float y = float((packedPos >> 11) & 2047u) / 2047.0;
            float z = float((packedPos >> 22) & 1023u) / 1023.0;

            vec3 norm = vec3(x, y, z);
            float depthZ = (norm.z / uScale.z) + uMin.z;
            vec3 pos = vec3(
                (norm.x / uScale.x) + uMin.x,
                (norm.y / uScale.y) + uMin.y,
                depthZ
            );

            vColor = vec3(float(rData), float(gData), float(bData)) / 255.0;

            vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
            gl_Position = projectionMatrix * mvPosition;

            float dist = -mvPosition.z; 
            float closeDist = 0.01;
            float farDist = 4.0;
            float t = clamp((dist - closeDist) / (farDist - closeDist), 0.0, 1.0);
            gl_PointSize = mix(1.0, 3.0, t);
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

const standardShader = {
    uniforms: {
        uMin: { value: new THREE.Vector3() },
        uScale: { value: new THREE.Vector3() },
        uBitsPos: { value: new THREE.Vector3(8, 8, 8) }, 
    },
    vertexShader: `
        precision highp int;
        precision highp float;

        in uint xData;
        in uint yData;
        in uint zData;
        in uint rData;
        in uint gData;
        in uint bData;

        uniform vec3 uMin;
        uniform vec3 uScale;
        uniform vec3 uBitsPos;

        out vec3 vColor;

        void main() {
            uint max_x = (1u << uint(uBitsPos.x)) - 1u;
            uint max_y = (1u << uint(uBitsPos.y)) - 1u;
            uint max_z = (1u << uint(uBitsPos.z)) - 1u;

            vec3 norm = vec3(
                float(xData) / float(max_x),
                float(yData) / float(max_y),
                float(zData) / float(max_z)
            );
            float invZ = (norm.z / uScale.z) + uMin.z;
            float depthZ = 1.0 / max(invZ, 1e-6);
            vec3 pos = vec3(
                (norm.x / uScale.x) + uMin.x,
                (norm.y / uScale.y) + uMin.y,
                depthZ
            );

            // MED/LOW now use planar RGB like HIGH.
            float r = float(rData) / 255.0;
            float g = float(gData) / 255.0;
            float b = float(bData) / 255.0;

            vColor = vec3(r, g, b);

            vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
            gl_Position = projectionMatrix * mvPosition;

            float dist = -mvPosition.z; 
            float closeDist = 0.01;
            float farDist = 4.0;
            float t = clamp((dist - closeDist) / (farDist - closeDist), 0.0, 1.0);
            gl_PointSize = mix(1.0, 7.0, t);
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

    const matHigh = new THREE.ShaderMaterial({
        uniforms: THREE.UniformsUtils.clone(highQShader.uniforms),
        vertexShader: highQShader.vertexShader,
        fragmentShader: highQShader.fragmentShader,
        glslVersion: THREE.GLSL3 
    });

    // MED now uses packed coordinates, so it gets a dedicated HIGH-like material.
    // This avoids sharing uMin/uScale uniforms with HIGH chunks.
    const matMedPacked = new THREE.ShaderMaterial({
        uniforms: THREE.UniformsUtils.clone(highQShader.uniforms),
        vertexShader: highQShader.vertexShader,
        fragmentShader: highQShader.fragmentShader,
        glslVersion: THREE.GLSL3
    });

    const matMed = new THREE.ShaderMaterial({
        uniforms: THREE.UniformsUtils.clone(standardShader.uniforms),
        vertexShader: standardShader.vertexShader,
        fragmentShader: standardShader.fragmentShader,
        glslVersion: THREE.GLSL3 
    });

    const matLow = new THREE.ShaderMaterial({
        uniforms: THREE.UniformsUtils.clone(standardShader.uniforms),
        vertexShader: standardShader.vertexShader,
        fragmentShader: standardShader.fragmentShader,
        glslVersion: THREE.GLSL3 
    });

    function createMesh() {
        const geo = new THREE.BufferGeometry();

        const mesh = new THREE.Points(geo, matLow); 
        mesh.frustumCulled = false;
        mesh.visible = false;
        mesh.rotateX(Math.PI); 
        scene.add(mesh);
        return mesh;
    }

    for (let i = 0; i < MAX_CHUNKS; i++) {
        meshPool.push(createMesh());
    }

    function updateAttribute(
        geo: THREE.BufferGeometry, 
        name: string, 
        sharedBuf: SharedArrayBuffer, 
        offset: number, 
        count: number, 
        stride: number
    ) {
        let newView;
        if (stride === 4) newView = new Uint32Array(sharedBuf, offset, count);
        else if (stride === 2) newView = new Uint16Array(sharedBuf, offset, count);
        else newView = new Uint8Array(sharedBuf, offset, count);

        let newAttr;
        if (stride === 4) newAttr = new THREE.Uint32BufferAttribute(newView, 1);
        else if (stride === 2) newAttr = new THREE.Uint16BufferAttribute(newView, 1);
        else newAttr = new THREE.Uint8BufferAttribute(newView, 1);

        (newAttr as any).gpuType = THREE.IntType; 
        geo.setAttribute(name, newAttr);
    }

    function updateMeshFromChunk(mesh: THREE.Points, sharedBuf: SharedArrayBuffer, offset: number) {
        const dv = new DataView(sharedBuf, offset, HEADER_SIZE);
        const geo = mesh.geometry;

        const minX = dv.getFloat32(0, true);
        const minY = dv.getFloat32(4, true);
        const minZ = dv.getFloat32(8, true);
        const scaleX = dv.getFloat32(12, true);
        const scaleY = dv.getFloat32(16, true);
        const scaleZ = dv.getFloat32(20, true);

        const mode = dv.getInt32(24, true);
        const numPoints = dv.getInt32(28, true);

        let currentOffset = offset + HEADER_SIZE;

        if (mode === MODE_HIGH || mode === MODE_MED) {
            // MED uses packed coordinates like HIGH, but keeps isolated transform uniforms.
            mesh.material = (mode === MODE_HIGH) ? matHigh : matMedPacked;
            const mat = mesh.material as THREE.ShaderMaterial;
            
            mat.uniforms.uMin.value.set(minX, minY, minZ);
            mat.uniforms.uScale.value.set(scaleX, scaleY, scaleZ);

            updateAttribute(geo, 'packedPos', sharedBuf, currentOffset, numPoints, 4);
            currentOffset += numPoints * 4;

            updateAttribute(geo, 'rData', sharedBuf, currentOffset, numPoints, 1);
            currentOffset += numPoints;
            currentOffset += (4 - (currentOffset % 4)) % 4; // Pad

            updateAttribute(geo, 'gData', sharedBuf, currentOffset, numPoints, 1);
            currentOffset += numPoints;
            currentOffset += (4 - (currentOffset % 4)) % 4; // Pad

            updateAttribute(geo, 'bData', sharedBuf, currentOffset, numPoints, 1);
        } else {
            // LOW keeps split XYZ bytes while MED moved to packed HIGH layout.
            let bx=8, by=8, bz=8;
            let sx=1;
            mesh.material = matLow;

            const mat = mesh.material as THREE.ShaderMaterial;
            mat.uniforms.uMin.value.set(minX, minY, minZ);
            mat.uniforms.uScale.value.set(scaleX, scaleY, scaleZ);
            mat.uniforms.uBitsPos.value.set(bx, by, bz);

            updateAttribute(geo, 'xData', sharedBuf, currentOffset, numPoints, sx);
            currentOffset += numPoints * sx;
            currentOffset += (4 - (currentOffset % 4)) % 4;

            updateAttribute(geo, 'yData', sharedBuf, currentOffset, numPoints, sx);
            currentOffset += numPoints * sx;
            currentOffset += (4 - (currentOffset % 4)) % 4;

            updateAttribute(geo, 'zData', sharedBuf, currentOffset, numPoints, sx);
            currentOffset += numPoints * sx;
            currentOffset += (4 - (currentOffset % 4)) % 4;

            updateAttribute(geo, 'rData', sharedBuf, currentOffset, numPoints, 1);
            currentOffset += numPoints;
            currentOffset += (4 - (currentOffset % 4)) % 4;

            updateAttribute(geo, 'gData', sharedBuf, currentOffset, numPoints, 1);
            currentOffset += numPoints;
            currentOffset += (4 - (currentOffset % 4)) % 4;

            updateAttribute(geo, 'bData', sharedBuf, currentOffset, numPoints, 1);
        }

        geo.setDrawRange(0, numPoints);
    }

    let pendingFrameId = -1;
    let accumulatedChunks: { offset: number; length: number }[] = [];

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

        // Accumulate chunks for the pending frame.
        accumulatedChunks.push(chunk);

        if (accumulatedChunks.length === numChunks) {
            // Commit the frame atomically after decoding all chunks.
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
