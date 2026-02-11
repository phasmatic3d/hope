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
            vec3 pos = (norm / uScale) + uMin;

            vColor = vec3(float(rData), float(gData), float(bData)) / 255.0;

            vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
            gl_Position = projectionMatrix * mvPosition;

            float safeDist = max(-mvPosition.z, 0.01);
            gl_PointSize = clamp((2.0 / safeDist), 1.0, 15.0);
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
        uBitsCol: { value: new THREE.Vector3(8, 8, 8) }
    },
    vertexShader: `
        precision highp int;
        precision highp float;

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
            uint max_x = (1u << uint(uBitsPos.x)) - 1u;
            uint max_y = (1u << uint(uBitsPos.y)) - 1u;
            uint max_z = (1u << uint(uBitsPos.z)) - 1u;

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

            float safeDist = max(-mvPosition.z, 0.01);
            gl_PointSize = clamp((2.0 / safeDist), 1.0, 15.0);
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

        if (mode === MODE_HIGH) {
            mesh.material = matHigh;
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
            // Match producer-side quantization profiles for MED and LOW payloads.
            let bx=11, by=11, bz=10;
            let br=8, bg=8, bb=8;
            let sx=2;
            let scol=4;

            if (mode === MODE_MED) {
                mesh.material = matMed;
            } else { // MODE_LOW
                mesh.material = matLow;
                bx=8; by=8; bz=8;
                sx=1;
            }

            const mat = mesh.material as THREE.ShaderMaterial;
            mat.uniforms.uMin.value.set(minX, minY, minZ);
            mat.uniforms.uScale.value.set(scaleX, scaleY, scaleZ);
            mat.uniforms.uBitsPos.value.set(bx, by, bz);
            mat.uniforms.uBitsCol.value.set(br, bg, bb);

            updateAttribute(geo, 'xData', sharedBuf, currentOffset, numPoints, sx);
            currentOffset += numPoints * sx;
            currentOffset += (4 - (currentOffset % 4)) % 4;

            updateAttribute(geo, 'yData', sharedBuf, currentOffset, numPoints, sx);
            currentOffset += numPoints * sx;
            currentOffset += (4 - (currentOffset % 4)) % 4;

            updateAttribute(geo, 'zData', sharedBuf, currentOffset, numPoints, sx);
            currentOffset += numPoints * sx;
            currentOffset += (4 - (currentOffset % 4)) % 4;

            updateAttribute(geo, 'packedColor', sharedBuf, currentOffset, numPoints, scol);
        }

        geo.setDrawRange(0, numPoints);
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