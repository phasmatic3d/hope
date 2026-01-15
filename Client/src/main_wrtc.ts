import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
//import { openConnection } from './transmissionWS';
import { openConnection } from './transmissionWebRTC';
import { FreeRoamController } from './FreeRoamController';
import { createGPUPointCloud } from './gpu_pc_renderer'; 

const SERVER_HEADER_SIZE = 6; // 1 (chunks) + 1 (id) + 4 (offset)

async function setupScene_gpu() {
    const POINT_BUDGET = 150_000;
    const MAX_BPP = 12; 

    const sharedEncodedBuffer = new SharedArrayBuffer(POINT_BUDGET * MAX_BPP * 2);
    const sharedEncodedView = new Uint8Array(sharedEncodedBuffer);

    const scene = new THREE.Scene();
    
    const grid = new THREE.GridHelper(10, 10);
    scene.add(grid);

    const renderer = new THREE.WebGLRenderer({
        antialias: false,
        powerPreference: "high-performance",
        precision: "highp",         
        depth: true,
        stencil: false,
        alpha: false });

    renderer.setPixelRatio(1.0);     
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.xr.enabled = true;
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;

    document.body.appendChild(renderer.domElement);
    document.body.appendChild(VRButton.createButton(renderer));

	const gl = renderer.getContext() as WebGL2RenderingContext;
    const ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
    const query = ext ? gl.createQuery() : null;

    const camCtrl = new FreeRoamController(scene, renderer, {
        startPosition: [0.0, 0.0, 1.0],
        baseSpeed: 1.0,
        sprintMultiplier: 2.0,
        damping: 10
    });
    
    const camera = camCtrl.camera;

    const processPointCloud = createGPUPointCloud(scene);

	let uploadTimeAccumulator = 0;
    let frameChunkCount = 0;

    function measureGPU() {
        if (!ext || !query) return;
        gl.beginQuery(ext.TIME_ELAPSED_EXT, query);
        renderer.render(scene, camera);
        gl.endQuery(ext.TIME_ELAPSED_EXT);
        requestAnimationFrame(checkQuery);
    }

    function checkQuery() {
        if (!ext || !query) return;
        
        const available = gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE);
        if (available) {
            const timeNs = gl.getQueryParameter(query, gl.QUERY_RESULT);
            const gpuTimeMs = timeNs / 1000000;
            
            if (Math.random() < 0.01) {
                console.log(`[Perf] GPU Render: ${gpuTimeMs.toFixed(3)}ms | CPU Upload: ${uploadTimeAccumulator.toFixed(3)}ms`);
            }
            
            uploadTimeAccumulator = 0; 
        } else {
            // Keep checking next frame
            requestAnimationFrame(checkQuery);
        }
    }

    renderer.setAnimationLoop(() => {
        camCtrl.update(0.01);
        //renderer.render(scene, camera);
        //return;
        if (ext) {
            measureGPU(); // Render wrapped in timer
        } else {
            renderer.render(scene, camera);
        }
    });

    // Resize
    window.addEventListener('resize', () => camCtrl.onResize(window.innerWidth, window.innerHeight));

    // --- 4. Network Logic ---
    const HALF_BUFFER_SIZE = sharedEncodedView.byteLength / 2;
    let bufferOffsetBase = 0; // Toggles between 0 and HALF_BUFFER_SIZE
    let lastFrameId = -1; // NEW: Track frame ID to handle lost start packets

    openConnection(
        (data: ArrayBuffer) => {
            const dv = new DataView(data);
            
            // Header: [Chunks (1)][FrameId (1)][ByteOffset (4)]
            const numChunks = dv.getUint8(0);
            const frameId = dv.getUint8(1);
            const chunkOffset = dv.getUint32(2, true); 

            // 1. Check for Late Packets (Wrap-around safe)
            // If frameId is significantly "older" than lastFrameId, drop it.
            const diff = (frameId - lastFrameId + 256) % 256;
            if (diff > 128 && lastFrameId !== -1) {
                // Packet is too old (from a previous loop of 255), ignore
                return;
            }

            // 2. Buffer Swapping (Triggered by ID change, not chunk index)
            if (frameId !== lastFrameId) {
                bufferOffsetBase = (bufferOffsetBase === 0) ? HALF_BUFFER_SIZE : 0;
                lastFrameId = frameId;
                uploadTimeAccumulator = 0;
            }

            // Copy Payload to Shared Memory
            // Payload starts after the 6-byte server header
            const payload = new Uint8Array(data, SERVER_HEADER_SIZE);
            const writePos = bufferOffsetBase + chunkOffset;
            
            // Safety Check
            if (writePos + payload.byteLength > sharedEncodedView.byteLength) {
                console.error(`Buffer Overflow writting at:${writePos} len:${payload.byteLength}`);
                return;
            }

            sharedEncodedView.set(payload, writePos);

            // Trigger Renderer
            // We pass the absolute offset in shared memory where this chunk lives
            const chunkUploadTime = processPointCloud(
                sharedEncodedBuffer,
                { offset: writePos, length: payload.byteLength },
                numChunks,
                frameId
            );

			uploadTimeAccumulator += chunkUploadTime!;
        },
        (err: unknown) => console.log('WebSocket Error:', err)
    );
}

//chrome://inspect/#devices
setupScene_gpu();