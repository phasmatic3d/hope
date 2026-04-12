import * as THREE from 'three';

const HEADER_SIZE = 32;

const MODE_HIGH = 0;
const MODE_MED = 1;
const MODE_LOW = 2;
const DRACO_MAGIC = [68, 82, 67, 79];




async function loadDracoDecoderModule() {
    try {
        const imported = await import('three/examples/jsm/libs/draco/draco_decoder.js');
        const draco = (imported as any).default ?? imported;
        const createDecoderModule = draco.createDecoderModule ?? draco;
        const module = await createDecoderModule({});
        console.info('[draco] decoder module loaded');
        return module;
    } catch (err) {
        console.error('[draco] failed to initialize decoder module', err);
        return null;
    }
}

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
            gl_PointSize = mix(10.0, 1.0, t);
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
            gl_PointSize = mix(10.0, 1.0, t);
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


const dracoShader = {
    vertexShader: `
        precision highp float;


        out vec3 vColor;

        void main() {
            vColor = color;
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            gl_Position = projectionMatrix * mvPosition;
            float dist = -mvPosition.z;
            float closeDist = 0.01;
            float farDist = 4.0;
            float t = clamp((dist - closeDist) / (farDist - closeDist), 0.0, 1.0);
            gl_PointSize = mix(10.0, 1.0, t);
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
    const matDraco = new THREE.ShaderMaterial({
        vertexShader: dracoShader.vertexShader,
        fragmentShader: dracoShader.fragmentShader,
        glslVersion: THREE.GLSL3,
        vertexColors: true
    });
    let dracoDecoderModule: any = null;
    let dracoUnavailableLogged = false;
    loadDracoDecoderModule().then((m: any) => { dracoDecoderModule = m; });

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

    function decodeDracoPayload(payload: Uint8Array) {
        if (!dracoDecoderModule) {
            if (!dracoUnavailableLogged) {
                console.warn('[draco] payload received before decoder became available');
                dracoUnavailableLogged = true;
            }
            return null;
        }
        const dracoPayloadSize = new DataView(payload.buffer, payload.byteOffset + 9, 4).getUint32(0, true);
        const dracoBytes = payload.subarray(13, 13 + dracoPayloadSize);
        const decoder = new dracoDecoderModule.Decoder();
        const decoderBuffer = new dracoDecoderModule.DecoderBuffer();
        decoderBuffer.Init(new Int8Array(dracoBytes.buffer, dracoBytes.byteOffset, dracoBytes.byteLength), dracoBytes.byteLength);
        const pc = new dracoDecoderModule.PointCloud();
        const status = decoder.DecodeBufferToPointCloud(decoderBuffer, pc);
        if (!status.ok() || pc.ptr === 0) {
            dracoDecoderModule.destroy(decoderBuffer);
            dracoDecoderModule.destroy(decoder);
            dracoDecoderModule.destroy(pc);
            return null;
        }
        const numPoints = pc.num_points();
        const posId = decoder.GetAttributeId(pc, dracoDecoderModule.POSITION);
        const colId = decoder.GetAttributeId(pc, dracoDecoderModule.COLOR);
        const posAttr = decoder.GetAttribute(pc, posId);
        const colAttr = colId >= 0 ? decoder.GetAttribute(pc, colId) : null;

        const posArray = new dracoDecoderModule.DracoFloat32Array();
        decoder.GetAttributeFloatForAllPoints(pc, posAttr, posArray);
        const positions = new Float32Array(numPoints * 3);
        for (let i = 0; i < positions.length; i++) positions[i] = posArray.GetValue(i);

        const colors = new Uint8Array(numPoints * 3);
        if (colAttr) {
            const colArray = new dracoDecoderModule.DracoUInt8Array();
            decoder.GetAttributeUInt8ForAllPoints(pc, colAttr, colArray);
            for (let i = 0; i < colors.length; i++) colors[i] = colArray.GetValue(i);
            dracoDecoderModule.destroy(colArray);
        }

        dracoDecoderModule.destroy(posArray);
        dracoDecoderModule.destroy(decoderBuffer);
        dracoDecoderModule.destroy(decoder);
        dracoDecoderModule.destroy(pc);
        return { positions, colors, numPoints };
    }

    function updateMeshFromDraco(mesh: THREE.Points, positions: Float32Array, colors: Uint8Array, numPoints: number) {
        const geo = mesh.geometry;
        mesh.material = matDraco;
        geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        const colorAttr = new THREE.Uint8BufferAttribute(colors, 3, true);
        geo.setAttribute('color', colorAttr);
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
            const hasDracoChunk = accumulatedChunks.some((ch) => {
                const payload = new Uint8Array(sharedBuf, ch.offset, ch.length);
                return payload[0] === DRACO_MAGIC[0] && payload[1] === DRACO_MAGIC[1] && payload[2] === DRACO_MAGIC[2] && payload[3] === DRACO_MAGIC[3];
            });
            if (hasDracoChunk && !dracoDecoderModule) {
                return performance.now() - tStart;
            }
            // Commit the frame atomically after decoding all chunks.
            for (let m of meshPool) m.visible = false;

            for (let i = 0; i < accumulatedChunks.length; i++) {
                if (i >= MAX_CHUNKS) break;

                const ch = accumulatedChunks[i];
                const mesh = meshPool[i];
                const payload = new Uint8Array(sharedBuf, ch.offset, ch.length);
                const isDraco = payload[0] === DRACO_MAGIC[0] && payload[1] === DRACO_MAGIC[1] && payload[2] === DRACO_MAGIC[2] && payload[3] === DRACO_MAGIC[3];
                if (isDraco) {
                    const decoded = decodeDracoPayload(payload);
                    if (!decoded) continue;
                    updateMeshFromDraco(mesh, decoded.positions, decoded.colors, decoded.numPoints);
                } else {
                    updateMeshFromChunk(mesh, sharedBuf, ch.offset);
                }
                mesh.visible = true;
            }

            accumulatedChunks = [];
        }

        return performance.now() - tStart;
    }
}
