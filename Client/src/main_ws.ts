import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
import { openConnection } from './transmissionWS';
import { FreeRoamController } from './FreeRoamController';
import { createGPUPointCloud } from './gpu_pc_renderer'; 
import { createPerfCsvExporter } from './perf_csv_export';

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
    const gpuQueries: WebGLQuery[] = [];

    // Mobile browsers may miss timer-query support, so fallback timing stays active.
    if (!ext) console.log('[Perf] EXT_disjoint_timer_query_webgl2 unavailable; using CPU render timing fallback.');

    const camCtrl = new FreeRoamController(scene, renderer, {
        startPosition: [0.0, 0.0, 1.0],
        baseSpeed: 1.0,
        sprintMultiplier: 2.0,
        damping: 10
    });

    const camera = camCtrl.camera;

    const processPointCloud = createGPUPointCloud(scene);
    const perfExporter = createPerfCsvExporter('point-cloud-perf-ws');

    // Queue decode-complete frames until one GPU render sample is ready for each.
    const pendingFrames: Array<{ frameId: number; decodeMs: number }> = [];
	let uploadTimeAccumulator = 0;
    let activeFrameId = -1;
    let receivedChunks = 0;
    let expectedChunks = 0;


    /**
     * Pairs one decode-complete frame with one render timing sample.
     * This keeps CSV rows aligned with fully assembled point clouds.
     */
    function recordCompletedFrame(renderMs: number) {
        const completed = pendingFrames.shift();
        if (!completed) return;

        perfExporter.addSample(completed.frameId, completed.decodeMs, renderMs);
    }

    /**
     * Starts one GPU timer query for the current render pass.
     * A fresh query per frame avoids re-reading stale timings from reused query objects.
     */
    function measureGPU() {
        if (!ext) return;

        const query = gl.createQuery();
        if (!query) {
            renderer.render(scene, camera);
            return;
        }

        gpuQueries.push(query);
        gl.beginQuery(ext.TIME_ELAPSED_EXT, query);
        renderer.render(scene, camera);
        gl.endQuery(ext.TIME_ELAPSED_EXT);
        pollCompletedQueries();
    }

    /**
     * Drains ready GPU timing queries in FIFO order.
     * This keeps timings aligned with render completion order.
     */
    function pollCompletedQueries() {
        if (!ext) return;

        while (gpuQueries.length > 0) {
            const query = gpuQueries[0];
            const available = gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE);
            if (!available) break;

            const isDisjoint = gl.getParameter(ext.GPU_DISJOINT_EXT);
            if (!isDisjoint) {
                const timeNs = gl.getQueryParameter(query, gl.QUERY_RESULT);
                const gpuTimeMs = timeNs / 1000000;

                // Timer query path on desktops uses true GPU elapsed time.
                recordCompletedFrame(gpuTimeMs);
                uploadTimeAccumulator = 0;
            }

            gl.deleteQuery(query);
            gpuQueries.shift();
        }
    }

    renderer.setAnimationLoop(() => {
        camCtrl.update(0.01);

        if (ext) {
            measureGPU(); // Render wrapped in GPU timer query.
        } else {
            // Fallback path uses CPU-side render timing on platforms without timer queries.
            const renderStart = performance.now();
            renderer.render(scene, camera);
            const cpuRenderMs = performance.now() - renderStart;
            recordCompletedFrame(cpuRenderMs);
        }
    });

    // Resize
    window.addEventListener('resize', () => camCtrl.onResize(window.innerWidth, window.innerHeight));

    // Download button gives a manual export path to normal disk.
    const downloadBtn = document.createElement('button');
    downloadBtn.textContent = 'Download Perf CSV';
    downloadBtn.style.position = 'fixed';
    downloadBtn.style.top = '12px';
    downloadBtn.style.right = '12px';
    downloadBtn.style.zIndex = '9999';
    downloadBtn.style.padding = '8px 12px';
    downloadBtn.style.borderRadius = '6px';
    downloadBtn.style.border = '1px solid #444';
    downloadBtn.style.background = '#ffffff';
    downloadBtn.style.color = '#111111';
    downloadBtn.style.cursor = 'pointer';
    // Manual export writes the current in-memory sample set to CSV.
    downloadBtn.addEventListener('click', () => {
        perfExporter.downloadCsvToDisk();
    });
    document.body.appendChild(downloadBtn);


    // --- 4. Network Logic ---
    const HALF_BUFFER_SIZE = sharedEncodedView.byteLength / 2;
    let bufferOffsetBase = 0; // Toggles between 0 and HALF_BUFFER_SIZE

    openConnection(
        (data: ArrayBuffer) => {
            const dv = new DataView(data);

            // Header: [Chunks (1)][FrameId (1)][ByteOffset (4)]
            const numChunks = dv.getUint8(0);
            const frameId = dv.getUint8(1);
            const chunkOffset = dv.getUint32(2, true);

            // Reset counters when a new frame id appears in the stream.
            if (frameId !== activeFrameId) {
                activeFrameId = frameId;
                receivedChunks = 0;
                expectedChunks = numChunks;
                uploadTimeAccumulator = 0;
            }

            // Double Buffering Swap Logic
            // If this is the *first* chunk of a frame (offset 0), we decide which buffer half to use.
            // (Assuming chunk 0 arrives first. If UDP unordered, this logic needs a frame tracker,
            // but for WS usually ordered).
            if (chunkOffset === 0) {
                bufferOffsetBase = (bufferOffsetBase === 0) ? HALF_BUFFER_SIZE : 0;
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
            receivedChunks += 1;

            // Full decode time is the sum across all chunks in the frame.
            if (expectedChunks > 0 && receivedChunks >= expectedChunks) {
                pendingFrames.push({ frameId, decodeMs: uploadTimeAccumulator });
            }
        },
        (err: unknown) => console.log('WebSocket Error:', err)
    );
}

//chrome://inspect/#devices
setupScene_gpu();
