import * as THREE from 'three';
//import './style.css'

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const geometry = new THREE.BoxGeometry();
const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
const cube = new THREE.Mesh(geometry, material);
scene.add(cube);

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
scene.add(points);

camera.position.z = 15;

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
  renderer.render(scene, camera);
}

animate();


