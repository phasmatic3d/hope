// @ts-nocheck

import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
import { openConnection } from './transmissionWS';
import {DecoderMessage, createPointCloudResult} from './types';

const worker = new Worker(new URL('./worker.ts', import.meta.url), {  });
const ENCODED_HEADER = 5

let decoderModule: any;
let pointCloud: THREE.Points | null = null;
let pointCloudGeometry: THREE.BufferGeometry | null = null;
let __filename: "MAIN"

function createPointCloudProcessor(
	scene: THREE.Scene,
	decodedPosView: Float32Array,
	decodedColView: Uint8Array
) {
	let pointCloudGeometry: THREE.BufferGeometry | null = null;
	let pointCloud: THREE.Points | null = null;

	// State for IMPORTANCE/FULL mode
	let expectedChunks = 0;   // how many chunks we expect for the current frame/batch
	let decodedChunks  = 0;   // how many chunks have finished decoding
	let totalPoints    = 0;   // total points across decoded chunks in this batch

	// Make the Points object once
	function ensurePointCloud() {
		if (pointCloudGeometry) return;
		pointCloudGeometry = new THREE.BufferGeometry();
		const mat = new THREE.PointsMaterial({ vertexColors: true, size: 0.1, sizeAttenuation: false });
		pointCloud = new THREE.Points(pointCloudGeometry, mat);
		pointCloud.scale.set(20, 20, 20);
		pointCloud.rotateX(Math.PI);
		pointCloud.position.set(0, -10, 13);
		scene.add(pointCloud);
	}

	// One persistent worker message listener
	const onWorkerMessage = (ev: MessageEvent) => {
		const msg = ev.data;
		if (!msg || msg.type !== 'decoded') return;

		decodedChunks += 1;
		totalPoints   += (msg.numPoints ?? 0);

		// When we’ve decoded them all, update the geometry
		if (expectedChunks > 0 && decodedChunks === expectedChunks) {
			ensurePointCloud();

			pointCloudGeometry!.setAttribute(
				'position',
				new THREE.BufferAttribute(
					decodedPosView.subarray(0, totalPoints * 3),
					3
				)
			);
			pointCloudGeometry!.setAttribute(
				'color',
				new THREE.BufferAttribute(
					decodedColView.subarray(0, totalPoints * 3),
					3,
					true
				)
			);
			pointCloudGeometry!.attributes.position.needsUpdate = true;
			pointCloudGeometry!.attributes.color.needsUpdate    = true;

			// reset for next batch
			expectedChunks = 0;
			decodedChunks  = 0;
			totalPoints    = 0;
		}
	};

	// Attach listener once
	// @ts-ignore
	if (!(worker as any).__pcpHandlerAttached) {
		worker.addEventListener('message', onWorkerMessage);
		(worker as any).__pcpHandlerAttached = true;
	}

	// Synchronous per-message entry point
	return (
		sharedBuf: SharedArrayBuffer,
		chunk: { offset: number; length: number },
		bufferCount: number
	): void => {
		const encView = new Uint8Array(sharedBuf);

		// NONE mode
		if (bufferCount === 0) {
			const { offset, length } = chunk;
			const raw = encView.subarray(offset, offset + length);

			const numPoints = (raw.byteLength / 15) | 0;

			const posBytes = raw.slice(0, 4 * 3 * numPoints);
			const colBytes = raw.slice(4 * 3 * numPoints);

			const positions = new Float32Array(
				posBytes.buffer,
				posBytes.byteOffset,
				numPoints * 3
			);
			const colors = new Uint8Array(
				colBytes.buffer,
				colBytes.byteOffset,
				numPoints * 3
			);

			ensurePointCloud();
			pointCloudGeometry!.setAttribute('position', new THREE.BufferAttribute(positions, 3));
			pointCloudGeometry!.setAttribute('color',    new THREE.BufferAttribute(colors, 3, true));
			pointCloudGeometry!.attributes.position.needsUpdate = true;
			pointCloudGeometry!.attributes.color.needsUpdate    = true;
			return;
		}

		// IMPORTANCE/FULL mode
		if (expectedChunks === 0) {
			expectedChunks = bufferCount;
			decodedChunks  = 0;
			totalPoints    = 0;
		}

		worker.postMessage({
			type: 'decode',
			offset: chunk.offset,
			length: chunk.length
		});
	};
}


async function setupScene() {
	const POINT_BUDGET = 100_000_000;

	const sharedEncodedBuffer = new SharedArrayBuffer(POINT_BUDGET);
	const sharedEncodedView   = new Uint8Array(sharedEncodedBuffer);

	const decodedPosBuffer = new SharedArrayBuffer(POINT_BUDGET * 3 * 4);
	const decodedColBuffer = new SharedArrayBuffer(POINT_BUDGET * 3 * 1);

	const decodedPosView = new Float32Array(decodedPosBuffer);
	const decodedColView = new Uint8Array(decodedColBuffer);

	const scene    = new THREE.Scene();
	const camera   = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
	const renderer = new THREE.WebGLRenderer();
	renderer.setSize(window.innerWidth, window.innerHeight);
	renderer.xr.enabled = true;
	document.body.appendChild(renderer.domElement);
	document.body.appendChild(VRButton.createButton(renderer));

	camera.position.set(0, -10, 15);

	const processPointCloud = createPointCloudProcessor(
		scene,
		decodedPosView,
		decodedColView
	);

	function animate() {
		requestAnimationFrame(animate);
		renderer.render(scene, camera);
	}
	animate();

	worker.postMessage({
		type: 'init',
		sharedEncodedBuffer,
		decodedPosBuffer,
		decodedColBuffer
	});

	openConnection(
		(data: ArrayBuffer) => {
			const dv          = new DataView(data);
			const bufferCount = dv.getUint8(0);
			const offset      = dv.getUint32(1, true);

			const payload = new Uint8Array(data, ENCODED_HEADER);
			sharedEncodedView.set(payload, offset);

			processPointCloud(sharedEncodedBuffer, { offset, length: payload.byteLength }, bufferCount);
		},
		(err: unknown) => console.log(__filename, 'WebSocket error:', err)
	);
}

setupScene();