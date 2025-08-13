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

	return async( sharedBuf: SharedArrayBuffer,
		 incomingBuffers: { offset: number; length: number}[] ,
		 bufferCount: number)
		 : Promise<void> => {

		const encView     = new Uint8Array(sharedBuf);
		const firstChunk  = incomingBuffers[0];      // { offset: number; length: number }
		const { offset, length } = firstChunk;
		// ─── NONE mode (header===0): raw floats+bytes ───
		if (bufferCount === 0) {
			const raw = encView.subarray(offset, offset + length);
			const byteLen = raw.byteLength;

			// each point = 3×float32 (12 bytes) + 3×uint8 (3 bytes) = 15 bytes
			const numPoints = byteLen / 15;

			const posBytes = raw.slice(0, 4*3 * numPoints);
			const colBytes = raw.slice(4*3 * numPoints);

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


			if (!pointCloudGeometry) {
				pointCloudGeometry = new THREE.BufferGeometry();
				const mat = new THREE.PointsMaterial({ vertexColors: true, size: 0.1, sizeAttenuation: false });
				pointCloud = new THREE.Points(pointCloudGeometry, mat);
				pointCloud.scale.set(20,20,20);
				pointCloud.rotateX(Math.PI);
				pointCloud.position.set(0,-10,13);
				scene.add(pointCloud);
			}

			pointCloudGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
			pointCloudGeometry.setAttribute("color",    new THREE.BufferAttribute(colors,    3, true));
			pointCloudGeometry.attributes.position.needsUpdate = true;
			pointCloudGeometry.attributes.color.needsUpdate    = true;

		}
		// ─── IMPORTANCE / FULL mode ───
		const chunks: DecoderMessage[] = [];
		let totalPoints = 0;
		for (const { offset, length } of incomingBuffers) {
			// send decode + where to write
			const chunk: DecoderMessage = await new Promise<DecoderMessage>(resolve => {
				worker.onmessage = ev => resolve(ev.data);
				worker.postMessage({ 
				type: 'decode', 
				offset, 
				length
				});
			});
			chunks.push(chunk);
			totalPoints += chunk.numPoints;
		}

		// Merge all decoded chunks into single position/color buffers
		if (!pointCloudGeometry) {
			pointCloudGeometry = new THREE.BufferGeometry();
			const mat = new THREE.PointsMaterial({ vertexColors: true, size: 0.1, sizeAttenuation: false });
			pointCloud = new THREE.Points(pointCloudGeometry, mat);
			pointCloud.scale.set(20,20,20);
			pointCloud.rotateX(Math.PI);
			pointCloud.position.set(0,-10,13);
			scene.add(pointCloud);
		}

		// Update
		pointCloudGeometry.setAttribute(
		'position',
			new THREE.BufferAttribute(
				decodedPosView.subarray(0, totalPoints * 3),
				3
			)
		);
		pointCloudGeometry.setAttribute(
			'color',
			new THREE.BufferAttribute(
				decodedColView.subarray(0, totalPoints * 3),
				3,
				true
			)
		);
		pointCloudGeometry.attributes.position.needsUpdate = true;
		pointCloudGeometry.attributes.color.needsUpdate = true;

	};
}


async function setupScene(){

	const POINT_BUDGET = 100_000_000;

	const sharedEncodedBuffer = new SharedArrayBuffer(POINT_BUDGET);
	const sharedEncodedView   = new Uint8Array(sharedEncodedBuffer);

	const decodedPosBuffer = new SharedArrayBuffer(POINT_BUDGET * 3 * 4);
	const decodedColBuffer = new SharedArrayBuffer(POINT_BUDGET * 3 * 1);

	const decodedPosView = new Float32Array(decodedPosBuffer);
	const decodedColView = new Uint8Array(decodedColBuffer);

  	const scene    = new THREE.Scene();
  	const camera   = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
  	const renderer = new THREE.WebGLRenderer();
  	renderer.setSize(window.innerWidth, window.innerHeight);
  	renderer.xr.enabled = true;
  	document.body.appendChild(renderer.domElement);
  	document.body.appendChild(VRButton.createButton(renderer));

  	camera.position.set(0, -10, 15);

  	// ─── WebSocket + point-cloud pipeline ───
  	const processPointCloud = createPointCloudProcessor(
		scene,
		decodedPosView,
		decodedColView
	);

	function animate() {
	    requestAnimationFrame( animate );
		renderer.render( scene, camera );
	}

	animate();

	worker.postMessage({
		type: 'init',
		sharedEncodedBuffer,
		decodedPosBuffer,
		decodedColBuffer
	});

	let expectedChunks = 0;
	let incomingBuffers: { offset: number; length: number }[] = [];

	openConnection(
		async (data) => {
			const dv     = new DataView(data);
			const bufferCount  = dv.getUint8(0);
			const offset = dv.getUint32(1, true);


			console.log(__filename, "Received new chunk...");
	

    		// first message of a group tells us how many to expect
    		if (expectedChunks === 0) {
				expectedChunks = bufferCount;
				console.log(__filename, `Setting expectedChunks: ${expectedChunks}`);
			}

    		const payload = new Uint8Array(data, ENCODED_HEADER);
     		sharedEncodedView.set(payload, offset);

    		incomingBuffers.push({ offset, length: payload.byteLength });

			console.log(__filename, `Incoming Buffers Length: ${incomingBuffers.length}`);


			for (let i = 0; i < incomingBuffers.length ; i++){
				console.log(__filename, `Buffer length: ${incomingBuffers[i].length}`);
			}

			await processPointCloud(sharedEncodedBuffer, incomingBuffers, bufferCount);

			expectedChunks = 0;
    		incomingBuffers = [];

		},
		(err) => console.log(__filename, 'WebSocket error:', err)
  	);
}

setupScene();