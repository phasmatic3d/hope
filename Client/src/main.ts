// @ts-nocheck

//import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
import { openConnection } from './transmissionWS';
import {DecoderMessage, createPointCloudResult} from './types';
import DrawCallInspector from './draw-call-inspector/DrawCallInspector.min.js';

const worker = new Worker(new URL('./worker.ts', import.meta.url), {  });
const ENCODED_HEADER = 5

let decoderModule: any;
let pointCloud: THREE.Points | null = null;
let pointCloudGeometry: THREE.BufferGeometry | null = null;

let __filename = "MAIN";

function mergeBuffers(chunks: Array<{positions: Float32Array, colors: Uint8Array}>){
	const totalPos = chunks.reduce((sum, c) => sum + c.positions.length, 0);
  	const totalCol = chunks.reduce((sum, c) => sum + c.colors.length, 0);


	const merged_positions = new Float32Array(totalPos);
	const merged_colors = new Uint8Array(totalCol);

	let posOffset = 0, colOffset = 0;
  	for (const { positions: p, colors: c } of chunks) {
  	  	merged_positions.set(p, posOffset);
  	  	merged_colors.set(c, colOffset);
  	  	posOffset += p.length;
  	  	colOffset += c.length;
  	}

  	return { merged_positions, merged_colors };
}

function createPointCloudProcessor(scene: THREE.Scene) {

	let pointCloudGeometry: THREE.BufferGeometry | null = null;
	let pointCloud: THREE.Points | null = null;

	return async( sharedBuf: SharedArrayBuffer, incomingBuffers: { offset: number; length: number}[] , bufferCount: number): Promise<createPointCloudResult> => {
				// read the first header
		const encView = new Uint8Array(sharedBuf);
		const firstOffset = incomingBuffers[0].offset;

		// ─── NONE mode (header===0): raw floats+bytes ───
		if (bufferCount === 0) {

			const raw = firstOffset.slice(1);           // strip header
			const byteLen = raw.byteLength;

			// each point = 3×float32 (12 bytes) + 3×uint8 (3 bytes) = 15 bytes
			const numPoints = byteLen / 15;

			const posBytes = raw.slice(0, 4*3 * numPoints);
			const colBytes = raw.slice(4*3 * numPoints);

			const positions = new Float32Array(posBytes);
			const colors    = new Uint8Array(colBytes);

			const geomStart = performance.now();

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
			
			const lastSceneUpdate = performance.now();
			const totalGeomTime = performance.now() - geomStart;
			return {decodeTime: 0, geometryUploadTime: totalGeomTime, lastSceneUpdateTime: lastSceneUpdate, chunkDecodeTimes: [0,0]};
		}
		// ─── IMPORTANCE / FULL mode ───
		const chunks: DecoderMessage[] = [];
		for (const { offset, length }  of incomingBuffers) {
			const msg: DecoderMessage = await new Promise(resolve => {
				worker.onmessage = ev => resolve(ev.data);
				worker.postMessage({type:'decode', offset, length});
			});
			chunks.push(msg);
		}

		const chunkDecodeTimes = chunks.map(c => c.dracoDecodeTime);
		const totalDecodeTime: number = chunks.map((c:DecoderMessage) => c.dracoDecodeTime).reduce((sum: number, t: number) => sum + t, 0);

		// Merge all decoded chunks into single position/color buffers
		const geomStart = performance.now();
		const { merged_positions, merged_colors } = mergeBuffers(chunks);

		// On the very first call, set up the Three.js Points object
		if (!pointCloudGeometry) {
			pointCloudGeometry = new THREE.BufferGeometry();
			const material = new THREE.PointsMaterial({
				vertexColors: true,
				size: 0.1,
				sizeAttenuation: false,
			});
			pointCloud = new THREE.Points(pointCloudGeometry, material);

			// apply your scene-wide transforms once
			pointCloud.scale.set(20, 20, 20);
			pointCloud.rotateX(Math.PI);
			pointCloud.position.set(0, -10, 13);

			scene.add(pointCloud);
		}

		// Upload the merged buffers into the geometry
		pointCloudGeometry.setAttribute("position", new THREE.BufferAttribute(merged_positions, 3));
		pointCloudGeometry.setAttribute("color", new THREE.BufferAttribute(merged_colors, 3, true));
		pointCloudGeometry.attributes.position.needsUpdate = true;
		pointCloudGeometry.attributes.color.needsUpdate = true;

		const lastSceneUpdate = performance.now();

		const totalGeomTime = performance.now() - geomStart;

		return {decodeTime: totalDecodeTime, geometryUploadTime: totalGeomTime, lastSceneUpdateTime: lastSceneUpdate, chunkDecodeTimes: chunkDecodeTimes};
	};
}


async function setupScenePromise(){

  	const scene    = new THREE.Scene();
  	const camera   = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
  	const renderer = new THREE.WebGLRenderer();
  	renderer.setSize(window.innerWidth, window.innerHeight);
  	renderer.xr.enabled = true;
  	document.body.appendChild(renderer.domElement);
  	document.body.appendChild(VRButton.createButton(renderer));

  	camera.position.set(0, -10, 15);

  	// ─── WebSocket + point-cloud pipeline ───
  	const processPointCloud = createPointCloudProcessor(scene);
	const dci = new DrawCallInspector( renderer, scene, camera, {} );
	dci.mount();

	//renderer.setAnimationLoop(() => {
	//	statsFPS.begin();
	//	statsMS.begin();

	//	renderer.render(scene, camera);

	//	statsFPS.end();
	//	statsMS.end();
	//});

	function animate() {
	    requestAnimationFrame( animate );
		dci.update();
		dci.begin();
		renderer.render( scene, camera );
		dci.end();
	}

	animate();

	function waitForNextFrame(renderer: THREE.WebGLRenderer, since: number): Promise<number> {
		// Use XR session’s RAF when the headset is presenting, otherwise window RAF
		const xrSession = renderer.xr?.isPresenting
				? renderer.xr.getSession()!
				: null;

		const raf = xrSession
				? xrSession.requestAnimationFrame.bind(xrSession)
				: window.requestAnimationFrame;

		return new Promise<number>(resolve => {
			raf((t: DOMHighResTimeStamp /* or XRFrame timestamp */) => {
			resolve(t - since);          // ms elapsed until *presentation*
			});
		});
	}

	const POINT_BUDGET   = 500_000;  // in bytes
	const sharedEncodedBuffer = new SharedArrayBuffer(POINT_BUDGET);
	const sharedEncodedView   = new Uint8Array(sharedEncodedBuffer);

	worker.postMessage({ type: 'init', sharedBuf: sharedEncodedBuffer });

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

    		// if we haven’t got them all yet, bail out early
			if (incomingBuffers.length < expectedChunks) {
				console.log(__filename, "BAIL OUT THE SYSTEM IS GOING TO FREEZE");
				return {
					decodeTime: 0,
					geometryUploadTime: 0,
					frameTime: 0,
					totalTime: 0,
					chunkDecodeTimes: [0]
				};
			}

			console.log(__filename, `Incoming Buffers Length: ${incomingBuffers.length}`);


			for (let i = 0; i < incomingBuffers.length ; i++){
				console.log(__filename, `Buffer length: ${incomingBuffers[i].length}`);
			}

			const { decodeTime, chunkDecodeTimes, geometryUploadTime, lastSceneUpdateTime}
			 = await processPointCloud(sharedEncodedBuffer, incomingBuffers, bufferCount);

			expectedChunks = 0;
    		incomingBuffers = [];
			// this is not correct
			// const frameTime  = await waitForNextFrame(renderer, lastSceneUpdateTime);
			const frameTime = 0;
			const totalTime = frameTime + decodeTime + geometryUploadTime;
			// send timing metrics back to server here if needed
			console.log(
				__filename,
				`Per-chunk decode times: [${chunkDecodeTimes.join(", ")}] ms\n` +
    		  	`Worker decode: ${decodeTime} ms, ` +
    		  	`Geometry upload: ${geometryUploadTime} ms, ` +
				`Frame Render: ${frameTime} ms, ` + 
				`Total time: ${totalTime} ms,`
    		);

			return { decodeTime, geometryUploadTime, frameTime, totalTime, chunkDecodeTimes};

		},
		(err) => console.log(__filename, 'WebSocket error:', err)
  	);
}

async function setupScene() {
	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

	const renderer = new THREE.WebGLRenderer();
	renderer.setSize(window.innerWidth, window.innerHeight);
	renderer.xr.enabled = true;
	document.body.appendChild(renderer.domElement);
	document.body.appendChild(VRButton.createButton(renderer));

	loadAndUpdatePointCloudFromWS_worker(scene);


	camera.position.z = 15;
	camera.position.y = -10;

	//animate();
	renderer.setAnimationLoop(() => {
		
		renderer.render(scene, camera);
	});
}

async function loadAndUpdatePointCloudFromWS_worker(scene: THREE.Scene) {

	const pending: Array<{ positions: Float32Array; colors: Uint8Array; numPoints: number }> = [];
	let numPC = 0;

	worker.onmessage = (event: MessageEvent<{positions: Float32Array; colors: Uint8Array; numPoints: number}>) => {
    	pending.push({ positions: event.data.positions, colors: event.data.colors, numPoints: event.data.numPoints });
    	console.time("Rendering")
    	if (pending.length === numPC) {
    		// concatenate all N chunks
    		let totalPts = 0;
    		let totalCols = 0;
    		for (const c of pending) {
				totalPts += c.positions.length;
				totalCols += c.colors.length;
    		}

			const mergedPos = new Float32Array(totalPts);
			const mergedCol = new Uint8Array(totalCols);

			let posOffset = 0, colorOffset = 0; // offsets for the buffers
			for (const pc of pending) {
				mergedPos.set(pc.positions, posOffset);
				mergedCol.set(pc.colors, colorOffset);
				posOffset += pc.positions.length;
				colorOffset += pc.colors.length;
			}

			// update or create geometry
			if (pointCloudGeometry) {
				pointCloudGeometry.setAttribute('position', new THREE.BufferAttribute(mergedPos, 3));
				pointCloudGeometry.setAttribute('color',    new THREE.BufferAttribute(mergedCol, 3, true));
				pointCloudGeometry.attributes.position.needsUpdate = true;
				pointCloudGeometry.attributes.color.needsUpdate    = true;
			} else {
				pointCloudGeometry = new THREE.BufferGeometry();
				pointCloudGeometry.setAttribute('position', new THREE.BufferAttribute(mergedPos, 3));
				pointCloudGeometry.setAttribute('color',    new THREE.BufferAttribute(mergedCol, 3, true));
				const material = new THREE.PointsMaterial({
					vertexColors: true,
					size: 0.01,
					sizeAttenuation: false
				});
				pointCloud = new THREE.Points(pointCloudGeometry, material);
				pointCloud.scale.set(20, 20, 20); 
				pointCloud.rotateX(3.14) // HARDCODED ROTATION: TODO (AND ALL OF THE OTHER TRANSFORMATIONS)
				pointCloud.position.y = -10;
				pointCloud.position.z = 13;
				pointCloud.position.x = 0;
				scene.add(pointCloud);
			}

			numPC = 0;
			pending.length = 0;
			console.timeEnd("Rendering")
    	}
	};
  
  	/*openConnection(
		(data: ArrayBuffer) => {
 	 		// first byte tells us how many chunks to expect
 	 		const dv = new DataView(data);
 	 		const count = dv.getUint8(0);

 	 		if (numPC === 0) {
 	 		  	numPC = count;
 	 		}

 	 		// slice off that prefix and send the raw Draco bytes to the worker
 	 		const dracoBuf = data.slice(1);
 	 		worker.postMessage({ data: dracoBuf });
 	 	}, 
		(msg) => {
 	 	  	console.log("Reject", msg);
 	 	}
	);*/
}

//setupScene();
setupScenePromise();