import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
import { initDracoDecoder, decodePointCloud } from './dracoDecoder';
import { openConnection } from './transmissionWS';
import { PI, rand } from 'three/tsl';

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
  	const worker = new Worker(new URL('./worker.ts', import.meta.url));
  	let expectedChunks = 0;
  	const pendingChunks: Array<{ positions: Float32Array; colors: Uint8Array }> = [];

	worker.onmessage = (ev: MessageEvent<{ positions: Float32Array; colors: Uint8Array; numPoints: number }>) => {
		pendingChunks.push({ positions: ev.data.positions, colors: ev.data.colors });

		if (pendingChunks.length === expectedChunks) {
			// Merge all decoded chunks
			const { positions, colors } = mergeBuffers(pendingChunks);

			// Update or create the BufferGeometry
			if (!pointCloudGeometry) {
				pointCloudGeometry = new THREE.BufferGeometry();
				const material = new THREE.PointsMaterial({
					vertexColors: true,
					size: 3,
					sizeAttenuation: false
				});
				pointCloud = new THREE.Points(pointCloudGeometry, material);
				pointCloud.scale.set(5, -5, 5);
				pointCloud.position.set(0, -10, 13);
				scene.add(pointCloud);
			}

			pointCloudGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
			pointCloudGeometry.setAttribute('color',    new THREE.BufferAttribute(colors,   3, true));
			pointCloudGeometry.attributes.position.needsUpdate = true;
			pointCloudGeometry.attributes.color.needsUpdate    = true;

			pendingChunks.length = 0;      // reset for next message
		}
	}

	return async (rawData: ArrayBuffer) => {
		// First byte of rawData = how many chunks to expect
		expectedChunks = new DataView(rawData).getUint8(0);

		// Start high-res timer
		const start = performance.now();

		// Send compressed bytes to Worker
		worker.postMessage({ data: rawData.slice(1) }, [rawData.slice(1)]);

		// Wait until onmessage has merged them (polled via a tiny Promise)
		await new Promise<void>(resolve => {
			const check = () => (pendingChunks.length === 0 && expectedChunks === 0) ? resolve() : setTimeout(check, 1);
			check();
		});

		const duration = performance.now() - start;
		console.log(`Decode + geometry update: ${duration.toFixed(1)} ms`);
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
	// ─── Set up cube and random-walk points ───
  	const geometry = new THREE.BoxGeometry();
  	const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
  	const cube    = new THREE.Mesh(geometry, material);
  	//scene.add(cube);

  	const numPoints = 1000;
  	const positions = new Float32Array(numPoints * 3);
  	const colors = new Float32Array(numPoints * 3);
  	for (let i = 0; i < positions.length; i++) {
  	  	positions[i] = (Math.random() - 0.5) * 10;
  	  	colors[i] = Math.random();
  	}
  	const points_geometry = new THREE.BufferGeometry();
  	points_geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  	points_geometry.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
	const points_material = new THREE.PointsMaterial({
		//color: 0xff44aa,
		vertexColors: true, // Enable per-vertex colors
		size: 10.9,
		sizeAttenuation: false
	});

	const points = new THREE.Points(points_geometry, points_material);
	//scene.add(points);

	camera.position.z = 15;
	camera.position.y = -10;
  	//scene.add(randomPoints);

  	// ─── WebSocket + point-cloud pipeline ───
  	const processPointCloud = createPointCloudProcessor(scene);

	openConnection(
		async (data) => {
			const receivedAt = performance.now();
			await processPointCloud(data);
			const doneAt = performance.now();
			// send timing metrics back to server here if needed
			console.log(`Total client processing: ${(doneAt - receivedAt).toFixed(1)} ms`);
		},
		(err) => console.error('WebSocket error:', err)
  	);

	renderer.setAnimationLoop(() => {
		const positions = points_geometry.getAttribute('position') as THREE.BufferAttribute;
		const colors = points_geometry.getAttribute('color') as THREE.BufferAttribute;

		for (let i = 0; i < positions.count; i++) {
			const i3 = i * 3;

			// Example: Random walk
			positions.array[i3 + 0] += (Math.random() - 0.5) * 0.1;
			positions.array[i3 + 1] += (Math.random() - 0.5) * 0.1;
			positions.array[i3 + 2] += (Math.random() - 0.5) * 0.1;

			// Example: Change color over time
			/*colors.array[i3 + 0] = Math.random();
			colors.array[i3 + 1] = Math.random();
			colors.array[i3 + 2] = Math.random();*/
		}

		positions.needsUpdate = true;
		colors.needsUpdate = true;
		points_geometry.setDrawRange(0, 1000); // Start with 0 points
  
  
		cube.rotation.x += 0.01;
		cube.rotation.y += 0.01;
		points.rotation.y += 0.01;
		//pointCloud && (pointCloud.rotation.y += 0.01);
		renderer.render(scene, camera);
	});
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

	// Load and decode the Draco file
	//loadAndUpdatePointCloud(scene, 'bunny.drc');
	loadAndUpdatePointCloudFromWS_worker(scene);

	const geometry = new THREE.BoxGeometry();
	const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
	const cube = new THREE.Mesh(geometry, material);
	//scene.add(cube);

	// Generate some points
	const numPoints = 1000;
	const positions = new Float32Array(numPoints * 3);
	const colors = new Float32Array(numPoints * 3);
	for (let i = 0; i < numPoints * 3; i++) {
		positions[i] = (Math.random() - 0.5) * 10; // spread in 3D space
		colors[i] = Math.random(); // Red/Green/Blue
	}
	const points_geometry = new THREE.BufferGeometry();
	points_geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
	points_geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
	const points_material = new THREE.PointsMaterial({
		//color: 0xff44aa,
		vertexColors: true, // Enable per-vertex colors
		size: 10.9,
		sizeAttenuation: false
	});

	const points = new THREE.Points(points_geometry, points_material);
	//scene.add(points);

	camera.position.z = 15;
	camera.position.y = -10;

	//animate();
	renderer.setAnimationLoop(() => {
		const positions = points_geometry.getAttribute('position') as THREE.BufferAttribute;
		const colors = points_geometry.getAttribute('color') as THREE.BufferAttribute;

		for (let i = 0; i < positions.count; i++) {
			const i3 = i * 3;

			// Example: Random walk
			positions.array[i3 + 0] += (Math.random() - 0.5) * 0.1;
			positions.array[i3 + 1] += (Math.random() - 0.5) * 0.1;
			positions.array[i3 + 2] += (Math.random() - 0.5) * 0.1;

			// Example: Change color over time
			/*colors.array[i3 + 0] = Math.random();
			colors.array[i3 + 1] = Math.random();
			colors.array[i3 + 2] = Math.random();*/
		}

		positions.needsUpdate = true;
		colors.needsUpdate = true;
		points_geometry.setDrawRange(0, 1000); // Start with 0 points
  
  
		cube.rotation.x += 0.01;
		cube.rotation.y += 0.01;
		points.rotation.y += 0.01;
		//pointCloud && (pointCloud.rotation.y += 0.01);
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
					size: 3.0,
					sizeAttenuation: false
				});
				pointCloud = new THREE.Points(pointCloudGeometry, material);
				pointCloud.scale.set(5, -5, 5);
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
