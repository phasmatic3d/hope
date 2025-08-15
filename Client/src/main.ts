
import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
import { openConnection } from './transmissionWS';
import { DecoderMessage } from './types';
import { FreeRoamController } from './FreeRoamController';

const worker = new Worker(new URL('./worker.ts', import.meta.url), { });
const ENCODED_HEADER = 5; // Header bytes

function createPointCloudProcessor(
	scene: THREE.Scene,
	decodedPosView: Float32Array,
	decodedColView: Uint8Array
) {
	let pointCloudGeometry: THREE.BufferGeometry | null = null;
	let pointCloud: THREE.Points | null = null;

	// Frame state (importance/full mode)
	let expectedChunks = 0;	
	let decodedChunks  = 0;	
	let totalPoints    = 0;		
	let currentPointCursor = 0;	// next write start (in points)
	let inFlight = false;		// true while a chunk is being decoded
	const queue: { offset: number; length: number }[] = [];

	function ensurePointCloud() {
		if (pointCloudGeometry) return;
		pointCloudGeometry = new THREE.BufferGeometry();
		const mat = new THREE.PointsMaterial({ vertexColors: true, size: 0.1, sizeAttenuation: false });
		pointCloud = new THREE.Points(pointCloudGeometry, mat);
		pointCloud.rotateX(Math.PI);
		//pointCloud.frustumCulled = false;
		scene.add(pointCloud);
	}

	function startNext() {
		if (inFlight) return;
		if (queue.length === 0) return;
		const next = queue.shift()!;
		const writeIndex = currentPointCursor; // in points
		inFlight = true;
		worker.postMessage({
			type: 'decode',
			offset: next.offset,
			length: next.length,
			writeIndex
		});
	}

	// Persistent worker listener
	const onWorkerMessage = (ev: MessageEvent) => {
		const msg = ev.data as DecoderMessage | { type: 'error'; message: string };
		if (!msg) return;

		if ((msg as any).type === 'error') {
			console.error('Worker error:', (msg as any).message);
			// Fail-safe: end the frame so we don't get stuck
			expectedChunks = 0;
			decodedChunks  = 0;
			totalPoints    = 0;
			currentPointCursor = 0;
			inFlight = false;
			queue.length = 0;
			return;
		}

		if (msg.type !== 'decoded') return;

		// advance cursors
		decodedChunks += 1;
		totalPoints   += msg.numPoints;
		currentPointCursor += msg.numPoints;
		inFlight = false;

		if (decodedChunks < expectedChunks) {
			startNext();
			return;
		}

		if (expectedChunks > 0 && decodedChunks === expectedChunks) {
			ensurePointCloud();
			
			pointCloudGeometry!.setAttribute(
				'position',
				new THREE.BufferAttribute(decodedPosView.subarray(0, totalPoints * 3), 3)
			);
			pointCloudGeometry!.setAttribute(
				'color',
				new THREE.BufferAttribute(decodedColView.subarray(0, totalPoints * 3), 3, true)
			);

			pointCloudGeometry!.setDrawRange(0, totalPoints); // prevents leftover points
			pointCloudGeometry!.attributes.position.needsUpdate = true;
			pointCloudGeometry!.attributes.color.needsUpdate    = true;
			// Reset frame state
			expectedChunks = 0;
			decodedChunks  = 0;
			totalPoints    = 0;
			currentPointCursor = 0;
			inFlight = false;
			queue.length = 0;
		}
	};

	// Attach listener once
	// @ts-ignore
	if (!(worker as any).__pcpHandlerAttached) {
		worker.addEventListener('message', onWorkerMessage);
		(worker as any).__pcpHandlerAttached = true;
	}

	// Per-message entry point
	return (
		sharedBuf: SharedArrayBuffer,
		chunk: { offset: number; length: number },
		bufferCount: number
	): void => {
		const encView = new Uint8Array(sharedBuf);
		const HALF_BYTES = encView.length >>> 1;

		const isFrameStart = bufferCount > 0 && (chunk.offset % HALF_BYTES === 0);
		// NONE mode 
		if (bufferCount === 0) {
			const { offset, length } = chunk;
			const raw = encView.subarray(offset, offset + length);

			const numPoints = (raw.byteLength / 15) | 0;

			const posBytes = raw.slice(0, 4 * 3 * numPoints);
			const colBytes = raw.slice(4 * 3 * numPoints);

			const positions = new Float32Array(posBytes.buffer, posBytes.byteOffset, numPoints * 3);
			const colors    = new Uint8Array(  colBytes.buffer, colBytes.byteOffset, numPoints * 3);

			ensurePointCloud();
			pointCloudGeometry!.setAttribute('position', new THREE.BufferAttribute(positions, 3));
			pointCloudGeometry!.setAttribute('color',    new THREE.BufferAttribute(colors, 3, true));
			pointCloudGeometry!.setDrawRange(0, numPoints);
			pointCloudGeometry!.attributes.position.needsUpdate = true;
			pointCloudGeometry!.attributes.color.needsUpdate    = true;
			return;
		}

		// If a new frame starts while we were still in one, drop the old frame cleanly.
		if (isFrameStart && expectedChunks > 0 && (decodedChunks < expectedChunks || inFlight || queue.length)) {
			expectedChunks = 0;
			decodedChunks  = 0;
			totalPoints    = 0;
			currentPointCursor = 0;
			inFlight = false;
			queue.length = 0;
		}

		if (expectedChunks === 0) {
			expectedChunks = bufferCount;
			decodedChunks  = 0;
			totalPoints    = 0;
			currentPointCursor = 0;
			inFlight = false;
			queue.length = 0;
		}


		queue.push({ offset: chunk.offset, length: chunk.length });
		startNext();
	};
}

async function setupScene() {
	const POINT_BUDGET = 80_000_000;

	const sharedEncodedBuffer = new SharedArrayBuffer(POINT_BUDGET * 2);
	const sharedEncodedView   = new Uint8Array(sharedEncodedBuffer);

	const decodedPosBuffer = new SharedArrayBuffer(POINT_BUDGET * 3 * 4);
	const decodedColBuffer = new SharedArrayBuffer(POINT_BUDGET * 3 * 1);

	const decodedPosView = new Float32Array(decodedPosBuffer);
	const decodedColView = new Uint8Array(decodedColBuffer);

	const scene    = new THREE.Scene();
	const renderer = new THREE.WebGLRenderer();
	renderer.setSize(window.innerWidth, window.innerHeight);
	renderer.xr.enabled = true;
	document.body.appendChild(renderer.domElement);
	document.body.appendChild(VRButton.createButton(renderer));

	const camCtrl = new FreeRoamController(scene, renderer, {
		startPosition: [0.1, 0, 0.45],
		baseSpeed: 0.4,
		sprintMultiplier: 2.5,
		damping: 10
	});
	const camera = camCtrl.camera;

	const processPointCloud = createPointCloudProcessor(
		scene,
		decodedPosView,
		decodedColView
	);

	// animation loop
	const clock = new THREE.Clock();
	function animate() {
		const dt = clock.getDelta();
		camCtrl.update(dt); // mouse+WASD when not in VR

		//const pos = camCtrl.getPosition(new THREE.Vector3());
		//console.log(`cam pos: x=${pos.x.toFixed(3)} y=${pos.y.toFixed(3)} z=${pos.z.toFixed(3)}`);
		renderer.render(scene, camera);
		renderer.setAnimationLoop(animate); // VR-friendly
	}
	renderer.setAnimationLoop(animate);

	// window resize
	window.addEventListener('resize', () => camCtrl.onResize(window.innerWidth, window.innerHeight));

	worker.postMessage({
		type: 'init',
		sharedEncodedBuffer,
		decodedPosBuffer,
		decodedColBuffer
	});


	const HALF_BYTES = sharedEncodedView.byteLength >>> 1; // divide by 2
	let encodedBase = 0;                                   // 0 or HALF
	openConnection(
		(data: ArrayBuffer) => {
			const dv          = new DataView(data);
			const bufferCount = dv.getUint8(0);        // chunks in frame (0 = NONE mode)
			const offset      = dv.getUint32(1, true); // frame-relative offset

			if (bufferCount > 0 && offset === 0) {
			// flip to the other half so current decoding bytes can't be overwritten
			encodedBase = (encodedBase === 0) ? HALF_BYTES : 0;
			}

			// Copy payload into the active half at absolute offset
			const payload = new Uint8Array(data, ENCODED_HEADER);
			const absOffset = encodedBase + offset;
			sharedEncodedView.set(payload, absOffset);

			processPointCloud(
			sharedEncodedBuffer,
			{ offset: absOffset, length: payload.byteLength },
			bufferCount
			);
		},
		(err: unknown) => console.log('WebSocket error:', err)
	);
}

setupScene();