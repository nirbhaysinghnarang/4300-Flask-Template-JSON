// render_clusters.js - nonty2
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 50);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor('#f8f5f1');
renderer.shadowMap.enabled = true;

document.body.style.margin = 0;
document.body.style.overflow = 'hidden';
document.body.appendChild(renderer.domElement);

// Create the home button
const homeButton = document.createElement('a');
homeButton.href = '/';
homeButton.innerText = 'Back to Dashboard';
homeButton.style.position = 'absolute';
homeButton.style.top = '20px';
homeButton.style.left = '20px';
homeButton.style.padding = '10px 15px';
homeButton.style.backgroundColor = 'var(--secondary-color)';
homeButton.style.color = '#ffffff';
homeButton.style.borderRadius = '8px';
homeButton.style.textDecoration = 'none';
homeButton.style.fontFamily = "'Source Sans Pro', sans-serif";
homeButton.style.fontWeight = 'bold';
homeButton.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.15)';
homeButton.style.zIndex = '1000';
homeButton.style.transition = 'all 0.2s ease';

homeButton.addEventListener('mouseover', () => {
  homeButton.style.transform = 'translateY(-2px)';
  homeButton.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.2)';
});

homeButton.addEventListener('mouseout', () => {
  homeButton.style.transform = 'translateY(0)';
  homeButton.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.15)';
});

document.body.appendChild(homeButton);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.rotateSpeed = 0.5;
controls.zoomSpeed = 0.8;
controls.autoRotate = false;

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let clusterData = [];
let pointSpheres = [];

fetch('/get-clusters')
  .then(response => response.json())
  .then(data => {
    clusterData = data;
    addSpheresAndCentroids();
    animate();
  })
  .catch(error => console.error('Error fetching cluster data:', error));

function addSpheresAndCentroids() {
  const clusters = {};

  clusterData.forEach((event, index) => {
    const x = event.x * 10;
    const y = event.y * 10;
    const z = event.z * 10;
    const color = new THREE.Color();
    color.setStyle(getClusterColor(event.cluster));

    // Create a sphere for each point
    const sphereGeometry = new THREE.SphereGeometry(1, 12, 12);
    const sphereMaterial = new THREE.MeshBasicMaterial({ color });
    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    sphere.position.set(x, y, z);
    sphere.userData.index = index;
    pointSpheres.push(sphere);
    scene.add(sphere);

    if (!clusters[event.cluster]) clusters[event.cluster] = [];
    clusters[event.cluster].push(new THREE.Vector3(x, y, z));
  });

  Object.keys(clusters).forEach(clusterId => {
    const pointsArray = clusters[clusterId];
    const centroid = new THREE.Vector3();
    pointsArray.forEach(p => centroid.add(p));
    centroid.divideScalar(pointsArray.length);

    const sphereGeometry = new THREE.SphereGeometry(0.9, 16, 16);
    const sphereMaterial = new THREE.MeshPhysicalMaterial({
      color: '#ffffff'
    });
    
    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    sphere.position.copy(centroid);
    scene.add(sphere);

    pointsArray.forEach(p => {
      const lineGeometry = new THREE.BufferGeometry().setFromPoints([centroid, p]);
      const lineMaterial = new THREE.LineBasicMaterial({ color: getClusterColor(parseInt(clusterId)), opacity: 0.3, transparent: true });
      const line = new THREE.Line(lineGeometry, lineMaterial);
      scene.add(line);
    });
  });
}

function getClusterColor(clusterId) {
  const hue = (clusterId * 137.508) % 360; 
  return `hsl(${hue}, 65%, 55%)`;
}
function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function onMouseClick(event) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(pointSpheres);

  const infoPanel = document.getElementById('info-panel');
  if (intersects.length > 0) {
    const clickedObject = intersects[0].object;
    const index = clickedObject.userData.index;
    const eventData = clusterData[index];
    
    const targetPosition = clickedObject.position.clone();
    const startPosition = camera.position.clone();
    const startRotation = controls.target.clone();
    
    const duration = 500; 
    const startTime = Date.now();
    
    function zoomToPoint() {
      const now = Date.now();
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      const easeProgress = progress < 0.5 
        ? 2 * progress * progress 
        : 1 - Math.pow(-2 * progress + 2, 2) / 2;
      
      const newPosition = new THREE.Vector3().lerpVectors(
        startPosition,
        targetPosition.clone().add(new THREE.Vector3(0, 0, 35)), 
        easeProgress
      );
      camera.position.copy(newPosition);
      const newTarget = new THREE.Vector3().lerpVectors(
        startRotation,
        targetPosition,
        easeProgress
      );
      controls.target.copy(newTarget);
      controls.update();
      
      if (progress < 1) {
        requestAnimationFrame(zoomToPoint);
      }
    }
    
    zoomToPoint();
    displayEventInfo(eventData);
  } else if (infoPanel) {
    infoPanel.remove();
  }
}


function displayEventInfo(event) {
  let infoPanel = document.getElementById('info-panel');
  if (!infoPanel) {
    infoPanel = document.createElement('div');
    infoPanel.id = 'info-panel';
    infoPanel.style.position = 'absolute';
    infoPanel.style.top = '20px';
    infoPanel.style.right = '20px';
    infoPanel.style.backgroundColor = 'rgba(255,255,255,0.97)';
    infoPanel.style.padding = '20px';
    infoPanel.style.borderRadius = '12px';
    infoPanel.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.08)';
    infoPanel.style.maxWidth = '320px';
    infoPanel.style.fontFamily = "'Source Sans Pro', sans-serif";
    infoPanel.style.transition = 'all 0.3s ease';
    document.body.appendChild(infoPanel);
  }
  infoPanel.innerHTML = `
    <h3 style="margin: 0 0 10px; font-size: 20px; color: var(--secondary-color);">${event["Name of Incident"] || 'Unnamed Event'}</h3>
    <div style="font-size: 14px; line-height: 1.5;">
      <strong style="color: var(--primary-color); margin-bottom: 10px;" >Cluster: </strong> ${event.cluster_name}<br>

      <em style="color: #666;">${event.description || 'No description available.'}</em>
    </div>
  `;
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener('resize', onWindowResize, false);
window.addEventListener('click', onMouseClick, false);