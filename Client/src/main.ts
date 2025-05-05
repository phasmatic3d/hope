import * as THREE from 'three';
import { initDracoDecoder, decodePointCloud } from './dracoDecoder';
import { openConnetion } from './transmissionWS';

let decoderModule: any;
let pointCloud: THREE.Points | null = null;
let pointCloudGeometry: THREE.BufferGeometry | null = null;

const worker = new Worker(new URL('./worker.ts', import.meta.url), {  });

async function setupScene() 
{
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

  const renderer = new THREE.WebGLRenderer();
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

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

  function animate() {
    requestAnimationFrame(animate);
  
    {
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
    }
  
  
    cube.rotation.x += 0.01;
    cube.rotation.y += 0.01;
    points.rotation.y += 0.01;
    //pointCloud && (pointCloud.rotation.y += 0.01);
    renderer.render(scene, camera);
  }
  
  animate();
}

async function loadAndUpdatePointCloud(scene: THREE.Scene, drcUrl: string) {
  const response = await fetch(drcUrl);
  const arrayBuffer = await response.arrayBuffer();
  
  // Decode the point cloud data
  const positions = decodePointCloud(decoderModule, arrayBuffer);

  // Create/update the geometry
  if (pointCloudGeometry) {
    pointCloudGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pointCloudGeometry.attributes.position.needsUpdate = true; // Mark as needing update
  } else {
    pointCloudGeometry = new THREE.BufferGeometry();
    pointCloudGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Set up the material and create the point cloud object
    const material = new THREE.PointsMaterial({ color: 0x00ff00, size: 0.05 });
    pointCloud = new THREE.Points(pointCloudGeometry, material);
    pointCloud.scale.x = 100;
    pointCloud.scale.y = 100;
    pointCloud.scale.z = 100;

    pointCloud.position.y = -10

    // Add to the scene
    scene.add(pointCloud);
  }
}

async function loadAndUpdatePointCloudFromWS(scene: THREE.Scene) {
  
  openConnetion((data: ArrayBuffer) => {
    // Decode the point cloud data
    const positions = decodePointCloud(decoderModule, data);

    // Create/update the geometry
    if (pointCloudGeometry) {
      pointCloudGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      pointCloudGeometry.attributes.position.needsUpdate = true; // Mark as needing update
    } 
    else 
    {
      pointCloudGeometry = new THREE.BufferGeometry();
      pointCloudGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

      // Set up the material and create the point cloud object
      const material = new THREE.PointsMaterial({ color: 0x00ff00, size: 0.05 });
      pointCloud = new THREE.Points(pointCloudGeometry, material);
      /*pointCloud.scale.x = 100;
      pointCloud.scale.y = 100;
      pointCloud.scale.z = 100;*/

      pointCloud.position.y = -10

      // Add to the scene
      scene.add(pointCloud);
    }

  }, (msg) => {
    console.log("Reject", msg)
  })  
}

async function loadAndUpdatePointCloudFromWS_worker(scene: THREE.Scene) {

  worker.onmessage = (event: MessageEvent<Float32Array>) => {
    // Create/update the geometry
    const positions = event.data;
    if (pointCloudGeometry) {
      pointCloudGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      pointCloudGeometry.attributes.position.needsUpdate = true; // Mark as needing update
    } 
    else 
    {
      pointCloudGeometry = new THREE.BufferGeometry();
      pointCloudGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

      // Set up the material and create the point cloud object
      const material = new THREE.PointsMaterial({ color: 0x00ff00, size: 0.1 });
      pointCloud = new THREE.Points(pointCloudGeometry, material);
      /*pointCloud.scale.x = 100;
      pointCloud.scale.y = 100;
      pointCloud.scale.z = 100;*/

      pointCloud.position.y = -10

      // Add to the scene
      scene.add(pointCloud);
    }
  };
  
  openConnetion((data: ArrayBuffer) => {
    // Decode the point cloud data
    worker.postMessage({
      //decoderModule: decoderModule, 
      data: data});   

  }, (msg) => {
    console.log("Reject", msg)
  })  
}

setupScene();
