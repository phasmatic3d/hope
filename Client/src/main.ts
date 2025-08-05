import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
import { initDracoDecoder } from './dracoDecoder';
import { openConnection } from './transmissionWS';

let decoderModule: any;
let pointCloud: THREE.Points | null = null;
let pointCloudGeometry: THREE.BufferGeometry | null = null;
const worker = new Worker(new URL('./worker.ts', import.meta.url), {  });

function mergeBuffers(chunks: Array<{positions: Float32Array, colors: Uint8Array}>){
	const totalPos = chunks.reduce((sum, c) => sum + c.positions.length, 0);
  	const totalCol = chunks.reduce((sum, c) => sum + c.colors.length, 0);


	const positions = new Float32Array(totalPos);
	const colors = new Uint8Array(totalCol);

	let posOffset = 0, colOffset = 0;
  	for (const { positions: p, colors: c } of chunks) {
  	  	positions.set(p, posOffset);
  	  	colors.set(c, colOffset);
  	  	posOffset += p.length;
  	  	colOffset += c.length;
  	}

  	return { positions, colors };
}

function createPointCloudProcessor(scene: THREE.Scene) {
  	let expectedChunks = 0;
  	const pendingChunks: Array<{ positions: Float32Array; colors: Uint8Array; decodeTime: number;}> = [];

	worker.onmessage = (ev: MessageEvent<
		{ 
			positions: Float32Array; 
			colors: Uint8Array; 
			numPoints: number; 
			dracoDecodeTime: number;
		}
	>) => {

		const { positions, colors, numPoints, dracoDecodeTime } = ev.data;

    	pendingChunks.push({
    	  	positions,
    	  	colors,
    	  	decodeTime: dracoDecodeTime
    	});


		if (pendingChunks.length === expectedChunks) {
			const totalDecodeTime = pendingChunks.reduce((sum, c) => sum + c.decodeTime, 0);
			// Merge all decoded chunks
			const { positions, colors } = mergeBuffers(pendingChunks);

			// Update or create the BufferGeometry
			const geomStart = performance.now();
			if (!pointCloudGeometry) {
				pointCloudGeometry = new THREE.BufferGeometry();
				const material = new THREE.PointsMaterial({
					vertexColors: true,
					size: 0.1,
					sizeAttenuation: false
				});
				pointCloud = new THREE.Points(pointCloudGeometry, material);
				pointCloud.scale.set(20, 20, 20); 
				pointCloud.rotateX(Math.PI); // HARDCODED ROTATION: TODO (AND ALL OF THE OTHER TRANSFORMATIONS)
				pointCloud.position.y = -10;
				pointCloud.position.z = 13;
				pointCloud.position.x = 0;
				scene.add(pointCloud);
			}

			pointCloudGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
			pointCloudGeometry.setAttribute('color',    new THREE.BufferAttribute(colors,   3, true));
			pointCloudGeometry.attributes.position.needsUpdate = true;
			pointCloudGeometry.attributes.color.needsUpdate    = true;
			const totalGeomTime = performance.now() - geomStart;

			lastDecodeTime = totalDecodeTime;
      		lastGeometryTime = totalGeomTime;

			pendingChunks.length = 0;      // reset for next message
			expectedChunks = 0;
		}
	}

	let lastDecodeTime = 0;
  	let lastGeometryTime = 0;

	return async (rawData: ArrayBuffer) => {
		// First byte of rawData = how many chunks to expect
		expectedChunks = new DataView(rawData).getUint8(0);

		// Start high-res timer
		const start = performance.now();

		console.log("Posting message to worker");
		// Send compressed bytes to Worker
		worker.postMessage({ data: rawData.slice(1) }, [rawData.slice(1)]);

		// Wait until onmessage has merged them (polled via a tiny Promise)
		await new Promise<void>(resolve => {
			const check = () => (pendingChunks.length === 0 && expectedChunks === 0) ? resolve() : setTimeout(check, 1);
			check();
		});

		const duration = performance.now() - start;
		console.log(`Decode + geometry update: ${duration.toFixed(1)} ms`);

		// return the two timings
    	return {
    	  	decodeTime: lastDecodeTime,
    	  	geometryUploadTime: lastGeometryTime
    	};
  	};
};


async function setupScenePromise(){

  	const scene    = new THREE.Scene();
  	const camera   = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
  	const renderer = new THREE.WebGLRenderer();
  	renderer.setSize(window.innerWidth, window.innerHeight);
  	renderer.xr.enabled = true;
  	document.body.appendChild(renderer.domElement);
  	document.body.appendChild(VRButton.createButton(renderer));

  	camera.position.set(0, -10, 15);
  	// ─── Draco decoder init ───
  	decoderModule = await initDracoDecoder();

  	// ─── WebSocket + point-cloud pipeline ───
  	const processPointCloud = createPointCloudProcessor(scene);

	renderer.setAnimationLoop(() => {
		renderer.render(scene, camera);
	});

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

	openConnection(
		async (data) => {
			const { decodeTime, geometryUploadTime} = await processPointCloud(data);
			const doneAt = performance.now();
			const frameTime  = await waitForNextFrame(renderer, doneAt);
			const totalTime = frameTime + decodeTime + geometryUploadTime;
			// send timing metrics back to server here if needed
			console.log(
    		  	`Worker decode: ${decodeTime} ms, ` +
    		  	`Geometry upload: ${geometryUploadTime} ms, ` +
				`Frame Render: ${frameTime} ms, ` + 
				`Total time: ${totalTime} ms,`
    		);

			return { decodeTime, geometryUploadTime, frameTime, totalTime};

		},
		(err) => console.error('WebSocket error:', err)
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

	// Set up Draco decoder (once)
	decoderModule = await initDracoDecoder();

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
  
  	openConnection(
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
	);
}

//setupScene();
setupScenePromise();
