// camera/FreeRoamController.ts
// @ts-nocheck
import * as THREE from 'three';
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js';

export type FreeRoamOptions = {
	fov?: number;				
	near?: number;				
	far?: number;				
	baseSpeed?: number;			
	sprintMultiplier?: number;	
	damping?: number;			
	startPosition?: THREE.Vector3 | [number,number,number];
	pointerLockUI?: boolean;
};

export class FreeRoamController {
	public camera: THREE.PerspectiveCamera;
	public controls: PointerLockControls;

	private renderer: THREE.WebGLRenderer;
	private velocity = new THREE.Vector3();
	private direction = new THREE.Vector3();
	private move = { forward:false, backward:false, left:false, right:false, up:false, down:false };
	private baseSpeed: number;
	private sprintMult: number;
	private damping: number;
	private enabled = true;

	private blocker?: HTMLElement;

	constructor(scene: THREE.Scene, renderer: THREE.WebGLRenderer, opts: FreeRoamOptions = {}) {
		this.renderer = renderer;

		const fov  = opts.fov ?? 75;
		const near = opts.near ?? 0.1;
		const far  = opts.far ?? 2000;

		this.baseSpeed  = opts.baseSpeed ?? 10;
		this.sprintMult = opts.sprintMultiplier ?? 2.5;
		this.damping    = opts.damping ?? 10;

		this.camera = new THREE.PerspectiveCamera(fov, window.innerWidth / window.innerHeight, near, far);
		this.controls = new PointerLockControls(this.camera, renderer.domElement);
		scene.add(this.controls.getObject());

		// start position
		if (opts.startPosition instanceof THREE.Vector3) {
			this.controls.getObject().position.copy(opts.startPosition);
		} else if (Array.isArray(opts.startPosition)) {
			const [x,y,z] = opts.startPosition;
			this.controls.getObject().position.set(x,y,z);
		} else {
			this.controls.getObject().position.set(0, -10, 15);
		}

		// optional pointer-lock overlay
		if (opts.pointerLockUI ?? true) this.installOverlay();

		// inputs
		this.onKeyDown = this.onKeyDown.bind(this);
		this.onKeyUp   = this.onKeyUp.bind(this);
		window.addEventListener('keydown', this.onKeyDown);
		window.addEventListener('keyup',   this.onKeyUp);

		// prevent scroll on Space/arrows while locked
		window.addEventListener('keydown', (e) => {
			if (this.controls.isLocked && (e.code === 'Space' || e.code.startsWith('Arrow'))) e.preventDefault();
		}, { passive:false });
	}

	// ── public API ────────────────────────────────────────────────────────────
	update(dt: number) {
		if (!this.enabled) return;
		// pause controls while in VR
		if (this.renderer.xr?.isPresenting) return;

		if (!this.controls.isLocked) return;

		// exponential damping
		this.velocity.x -= this.velocity.x * this.damping * dt;
		this.velocity.y -= this.velocity.y * this.damping * dt;
		this.velocity.z -= this.velocity.z * this.damping * dt;

		// intent
		this.direction.set(0,0,0);
		if (this.move.forward)  this.direction.z -= 1;
		if (this.move.backward) this.direction.z += 1;
		if (this.move.left)     this.direction.x -= 1;
		if (this.move.right)    this.direction.x += 1;
		if (this.move.up)       this.direction.y += 1;
		if (this.move.down)     this.direction.y -= 1;
		if (this.direction.lengthSq() > 0) this.direction.normalize();

		const sprinting = (window as any).event?.shiftKey ?? false; 
		const speed = this.baseSpeed * (sprinting ? this.sprintMult : 1);
		const accel = speed * 20;

		this.velocity.z += (-this.direction.z) * accel * dt;
		this.velocity.x += ( this.direction.x) * accel * dt;
		this.velocity.y += ( this.direction.y) * accel * dt;

		this.controls.moveRight(this.velocity.x * dt);
		this.controls.moveForward(this.velocity.z * dt);
		this.controls.getObject().position.y += (this.velocity.y * dt);
	}

	setEnabled(v: boolean) { this.enabled = v; }
	lock() { this.controls.lock(); }
	unlock() { this.controls.unlock(); }
	isLocked() { return this.controls.isLocked; }

	setPosition(x: number, y: number, z: number) { this.controls.getObject().position.set(x,y,z); }
	getPosition(out = new THREE.Vector3()) { return out.copy(this.controls.getObject().position); }

	setSpeed(base: number, sprintMult?: number) {
		this.baseSpeed = base;
		if (sprintMult !== undefined) this.sprintMult = sprintMult;
	}

	onResize(width: number, height: number) {
		this.camera.aspect = width / height;
		this.camera.updateProjectionMatrix();
		this.renderer.setSize(width, height);
	}

	dispose() {
		window.removeEventListener('keydown', this.onKeyDown);
		window.removeEventListener('keyup', this.onKeyUp);
		if (this.blocker && this.blocker.parentNode) this.blocker.parentNode.removeChild(this.blocker);
	}

	// ── internals ────────────────────────────────────────────────────────────
	private onKeyDown(e: KeyboardEvent) {
		switch (e.code) {
			case 'KeyW': this.move.forward = true; break;
			case 'KeyS': this.move.backward = true; break;
			case 'KeyA': this.move.left = true; break;
			case 'KeyD': this.move.right = true; break;
			case 'Space': this.move.up = true; break;
			case 'ControlLeft':
			case 'ControlRight': this.move.down = true; break;
		}
	}
	private onKeyUp(e: KeyboardEvent) {
		switch (e.code) {
			case 'KeyW': this.move.forward = false; break;
			case 'KeyS': this.move.backward = false; break;
			case 'KeyA': this.move.left = false; break;
			case 'KeyD': this.move.right = false; break;
			case 'Space': this.move.up = false; break;
			case 'ControlLeft':
			case 'ControlRight': this.move.down = false; break;
		}
	}

	private installOverlay() {
		const blocker = document.createElement('div');
		const instructions = document.createElement('div');
		blocker.style.cssText = `
			position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
			background:rgba(0,0,0,.4);font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;
			color:#fff;z-index:9999;
		`;
		instructions.innerHTML = `
			<div style="text-align:center;line-height:1.6">
				<h2 style="margin:0 0 .5rem">Click to roam</h2>
				<p style="margin:0">Mouse: look | WASD: move | Space/Ctrl: up/down | Shift: sprint</p>
			</div>`;
		blocker.appendChild(instructions);
		document.body.appendChild(blocker);
		this.blocker = blocker;

		instructions.addEventListener('click', () => this.controls.lock());
		this.controls.addEventListener('lock', () => { blocker.style.display = 'none'; });
		this.controls.addEventListener('unlock', () => { blocker.style.display = ''; });
	}
}