
import * as THREE from 'three';
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js';

export type FreeRoamOptions = {
	fov?: number;				// default 75
	near?: number;				// default 0.1
	far?: number;				// default 2000
	baseSpeed?: number;			// m/s (default 10)
	sprintMultiplier?: number;	// default 2.5
	damping?: number;			// velocity damping (default 10)
	touchLookSensitivity?: number; // radians per pixel (default 0.0025)
	joystickDeadzone?: number;	// 0..1 (default 0.12)
	startPosition?: THREE.Vector3 | [number,number,number];
	pointerLockUI?: boolean;	// desktop click-to-roam overlay (default true)
};

export class FreeRoamController {
	public camera: THREE.PerspectiveCamera;

	// Desktop controls
	private plc?: PointerLockControls;

	// Mobile rig (Yaw/Pitch nodes)
	private yawObj?: THREE.Object3D;
	private pitchObj?: THREE.Object3D;

	private renderer: THREE.WebGLRenderer;
	private scene: THREE.Scene;
	private enabled = true;

	// Common movement state
	private velocity = new THREE.Vector3();
	private direction = new THREE.Vector3();
	private baseSpeed: number;
	private sprintMult: number;
	private damping: number;

	// Desktop keyboard state
	private moveKeys = { forward:false, backward:false, left:false, right:false, up:false, down:false };
	private shiftDown = false;

	// Mobile input state
	private isMobile = isTouchDevice();
	private touchLookSensitivity: number;
	private joystickDeadzone: number;

	private mobileUI?: {
		container: HTMLElement;
		left: HTMLElement;
		leftStick: HTMLElement;
		right: HTMLElement;
	};
	private joyActive = false;
	private joyStart = new THREE.Vector2();
	private joyCurr  = new THREE.Vector2();
	private lookActive = false;
	private lookPrev   = new THREE.Vector2();

	// Optional desktop overlay
	private blocker?: HTMLElement;

	constructor(scene: THREE.Scene, renderer: THREE.WebGLRenderer, opts: FreeRoamOptions = {}) {
		this.scene = scene;
		this.renderer = renderer;

		const fov  = opts.fov ?? 75;
		const near = opts.near ?? 0.1;
		const far  = opts.far ?? 2000;

		this.baseSpeed  = opts.baseSpeed ?? 1.5;
		this.sprintMult = opts.sprintMultiplier ?? 2.5;
		this.damping    = opts.damping ?? 10;

		this.touchLookSensitivity = opts.touchLookSensitivity ?? 0.0025;
		this.joystickDeadzone     = opts.joystickDeadzone ?? 0.12;

		const startPos = opts.startPosition
			? (Array.isArray(opts.startPosition)
				? new THREE.Vector3(...opts.startPosition)
				: opts.startPosition.clone())
			: new THREE.Vector3(0, -10, 15);

		this.camera = new THREE.PerspectiveCamera(fov, window.innerWidth / window.innerHeight, near, far);

		if (this.isMobile) {
			// Mobile: build a yaw/pitch rig the camera sits in.
			this.yawObj = new THREE.Object3D();
			this.pitchObj = new THREE.Object3D();
			this.yawObj.add(this.pitchObj);
			this.pitchObj.add(this.camera);
			this.scene.add(this.yawObj);
			this.installMobileUI();
			this.installMobileEvents();
			this.setPosition(startPos.x, startPos.y, startPos.z);
		} else {
			// Desktop: pointer lock + keyboard
			this.plc = new PointerLockControls(this.camera, this.renderer.domElement);
			this.scene.add(this.plc.object);
			this.installDesktopOverlay(opts.pointerLockUI ?? true);
			this.installDesktopKeyboard();
			this.setPosition(startPos.x, startPos.y, startPos.z);
		}
	}

	// ───────────────────────────────── PUBLIC API ─────────────────────────────

	update(dt: number) {
		if (!this.enabled) return;
		if (this.renderer.xr?.isPresenting) return; // let VR drive camera

		// Apply damping to all axes
		this.velocity.x -= this.velocity.x * this.damping * dt;
		this.velocity.y -= this.velocity.y * this.damping * dt;
		this.velocity.z -= this.velocity.z * this.damping * dt;

		// Build intent vector (from keyboard or virtual joystick)
		if (this.isMobile) {
			const joy = this.getJoystickDir();
			this.direction.set(joy.x, 0, -joy.y); 

		} else {
			this.direction.set(0,0,0);
			if (this.moveKeys.forward)  this.direction.z -= 1;
			if (this.moveKeys.backward) this.direction.z += 1;
			if (this.moveKeys.left)     this.direction.x -= 1;
			if (this.moveKeys.right)    this.direction.x += 1;
			if (this.moveKeys.up)       this.direction.y += 1;
			if (this.moveKeys.down)     this.direction.y -= 1;
			if (this.direction.lengthSq() > 0) this.direction.normalize();
		}

		const sprinting = this.shiftDown; // desktop only (safe on mobile: false)
		const speed = this.baseSpeed * (sprinting ? this.sprintMult : 1);
		const accel = speed * 20;

		// Integrate in local camera space for X/Z; Y uses world up
		this.velocity.z += (-this.direction.z) * accel * dt;
		this.velocity.x += ( this.direction.x) * accel * dt;
		this.velocity.y += ( this.direction.y) * accel * dt;

		if (this.isMobile) {
			// move relative to yaw heading
			const yaw = this.yawObj!;
			// Right (x) and forward (z) in yaw space
			const right = new THREE.Vector3(1,0,0).applyQuaternion(yaw.quaternion);
			const fwd   = new THREE.Vector3(0,0,-1).applyQuaternion(yaw.quaternion);
			yaw.position.addScaledVector(right, this.velocity.x * dt);
			yaw.position.addScaledVector(fwd,   this.velocity.z * dt);
			yaw.position.y += this.velocity.y * dt;
		} else {
			// Desktop pointer-lock helpers
			this.plc!.moveRight(this.velocity.x * dt);
			this.plc!.moveForward(this.velocity.z * dt);
			this.plc!.object.position.y += (this.velocity.y * dt);
		}
	}

	onResize(width: number, height: number) {
		this.camera.aspect = width / height;
		this.camera.updateProjectionMatrix();
		this.renderer.setSize(width, height);
	}

	setEnabled(v: boolean) { this.enabled = v; }
	lock()   { if (!this.isMobile) this.plc?.lock(); }
	unlock() { if (!this.isMobile) this.plc?.unlock(); }
	isLocked() { return this.isMobile ? true : !!this.plc?.isLocked; }

	setPosition(x: number, y: number, z: number) {
		if (this.isMobile) this.yawObj!.position.set(x,y,z);
		else this.plc!.getObject().position.set(x,y,z);
	}
	getPosition(out = new THREE.Vector3()) {
		if (this.isMobile) return out.copy(this.yawObj!.position);
		return out.copy(this.plc!.getObject().position);
	}

	setSpeed(base: number, sprintMult?: number) {
		this.baseSpeed = base;
		if (sprintMult !== undefined) this.sprintMult = sprintMult;
	}

	dispose() {
		if (this.isMobile) this.removeMobileEvents();
		else this.removeDesktopKeyboard();
		if (this.blocker && this.blocker.parentNode) this.blocker.parentNode.removeChild(this.blocker);
		if (this.mobileUI?.container?.parentNode) this.mobileUI.container.parentNode.removeChild(this.mobileUI.container);
	}

	// ────────────────────────────── DESKTOP (mouse+kb) ────────────────────────

	private installDesktopOverlay(enable: boolean) {
		if (!enable) return;
		const blocker = document.createElement('div');
		const instructions = document.createElement('div');
		blocker.style.cssText = `
			position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
			background:rgba(0,0,0,.45);font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;
			color:#fff;z-index:9999;
		`;
		instructions.innerHTML = `
			<div style="text-align:center;line-height:1.6">
				<h2 style="margin:0 0 .5rem">Click here to move</h2>
				<p style="margin:0">Mouse: look | WASD: move | Space/Ctrl: up/down | Shift: sprint</p>
			</div>`;
		blocker.appendChild(instructions);
		document.body.appendChild(blocker);
		this.blocker = blocker;

		instructions.addEventListener('click', () => this.plc?.lock());
		this.plc!.addEventListener('lock',   () => { blocker.style.display = 'none'; });
		this.plc!.addEventListener('unlock', () => { blocker.style.display = '';   });
	}

	private installDesktopKeyboard() {
		this.onKeyDown = this.onKeyDown.bind(this);
		this.onKeyUp   = this.onKeyUp.bind(this);
		window.addEventListener('keydown', this.onKeyDown);
		window.addEventListener('keyup',   this.onKeyUp);

		// prevent scroll while locked
		window.addEventListener('keydown', (e) => {
			if (this.plc?.isLocked && (e.code === 'Space' || e.code.startsWith('Arrow'))) e.preventDefault();
		}, { passive:false });
	}
	private removeDesktopKeyboard() {
		window.removeEventListener('keydown', this.onKeyDown);
		window.removeEventListener('keyup',   this.onKeyUp);
	}
	private onKeyDown = (e: KeyboardEvent) => {
		switch (e.code) {
			case 'KeyW': this.moveKeys.forward = true; break;
			case 'KeyS': this.moveKeys.backward = true; break;
			case 'KeyA': this.moveKeys.left = true; break;
			case 'KeyD': this.moveKeys.right = true; break;
			case 'Space': this.moveKeys.up = true; break;
			case 'ControlLeft':
			case 'ControlRight': this.moveKeys.down = true; break;
			case 'ShiftLeft':
			case 'ShiftRight': this.shiftDown = true; break;
		}
	};
	private onKeyUp = (e: KeyboardEvent) => {
		switch (e.code) {
			case 'KeyW': this.moveKeys.forward = false; break;
			case 'KeyS': this.moveKeys.backward = false; break;
			case 'KeyA': this.moveKeys.left = false; break;
			case 'KeyD': this.moveKeys.right = false; break;
			case 'Space': this.moveKeys.up = false; break;
			case 'ControlLeft':
			case 'ControlRight': this.moveKeys.down = false; break;
			case 'ShiftLeft':
			case 'ShiftRight': this.shiftDown = false; break;
		}
	};

	// ──────────────────────────────── MOBILE (touch) ──────────────────────────

	private installMobileUI() {
		// translucent full-screen container with two zones: left (joystick) and right (look)
		const container = document.createElement('div');
		container.style.cssText = `
			position:fixed;inset:0;z-index:9999;pointer-events:auto;
			touch-action:none; /* we fully manage touch */
		`;

		const left = document.createElement('div');
		left.style.cssText = `
			position:absolute;left:0;top:0;bottom:0;width:45%;
		`;
		const right = document.createElement('div');
		right.style.cssText = `
			position:absolute;right:0;top:0;bottom:0;width:55%;
		`;

		// Joystick visuals
		const stickBase = document.createElement('div');
		stickBase.style.cssText = `
			position:absolute;width:120px;height:120px;border-radius:50%;
			background:rgba(255,255,255,.08);border:2px solid rgba(255,255,255,.2);
			left:20px;bottom:20px;transform:translate3d(0,0,0);
		`;
		const stick = document.createElement('div');
		stick.style.cssText = `
			position:absolute;width:64px;height:64px;border-radius:50%;
			background:rgba(255,255,255,.15);border:2px solid rgba(255,255,255,.3);
			left:20px;bottom:20px;transform:translate3d(28px,28px,0);
		`;
		left.appendChild(stickBase);
		left.appendChild(stick);

		container.appendChild(left);
		container.appendChild(right);
		document.body.appendChild(container);

		this.mobileUI = { container, left, leftStick: stick, right };
	}

	private installMobileEvents() {
		const left  = this.mobileUI!.left;
		const right = this.mobileUI!.right;
		const stick = this.mobileUI!.leftStick;

		// LEFT: joystick
		left.addEventListener('touchstart', (e) => {
			const t = e.changedTouches[0];
			this.joyActive = true;
			this.joyStart.set(t.clientX, t.clientY);
			this.joyCurr.copy(this.joyStart);
			e.preventDefault();
		}, { passive:false });

		left.addEventListener('touchmove', (e) => {
			if (!this.joyActive) return;
			const t = e.changedTouches[0];
			this.joyCurr.set(t.clientX, t.clientY);
			const delta = this.joyCurr.clone().sub(this.joyStart);
			const r = 50; // joystick radius in px around center (visual is 64 inside 120 base)
			const clamped = clampVec2Magnitude(delta, r);
			// Move the stick knob visually
			stick.style.transform = `translate3d(${28 + clamped.x}px, ${28 + clamped.y}px, 0)`;
			e.preventDefault();
		}, { passive:false });

		const endJoy = (e: TouchEvent) => {
			this.joyActive = false;
			this.joyStart.set(0,0);
			this.joyCurr.set(0,0);
			stick.style.transform = `translate3d(28px, 28px, 0)`;
			e.preventDefault();
		};
		left.addEventListener('touchend', endJoy, { passive:false });
		left.addEventListener('touchcancel', endJoy, { passive:false });

		// RIGHT: swipe to look
		right.addEventListener('touchstart', (e) => {
			const t = e.changedTouches[0];
			this.lookActive = true;
			this.lookPrev.set(t.clientX, t.clientY);
			e.preventDefault();
		}, { passive:false });

		right.addEventListener('touchmove', (e) => {
			if (!this.lookActive) return;
			const t = e.changedTouches[0];
			const dx = t.clientX - this.lookPrev.x;
			const dy = t.clientY - this.lookPrev.y;
			this.lookPrev.set(t.clientX, t.clientY);
			this.applyTouchLook(dx, dy);
			e.preventDefault();
		}, { passive:false });

		const endLook = (e: TouchEvent) => {
			this.lookActive = false;
			e.preventDefault();
		};
		right.addEventListener('touchend', endLook, { passive:false });
		right.addEventListener('touchcancel', endLook, { passive:false });
	}

	private removeMobileEvents() {
		// The UI container is removed in dispose(); listeners go away with it.
	}

	private getJoystickDir(): THREE.Vector2 {
		// Convert knob offset to normalized direction in range [-1,1] with deadzone
		if (!this.joyActive) return new THREE.Vector2(0,0);
		const delta = this.joyCurr.clone().sub(this.joyStart);
		const r = 50;
		const v = clampVec2Magnitude(delta, r).multiplyScalar(1 / r); // -1..1
		// Deadzone
		const dz = this.joystickDeadzone;
		const mag = v.length();
		if (mag < dz) return new THREE.Vector2(0,0);
		return v.multiplyScalar((mag - dz) / (1 - dz));
	}

	private applyTouchLook(dx: number, dy: number) {
		// Yaw (around world up) and pitch (local X) with clamp to avoid flip.
		const sens = this.touchLookSensitivity;
		const yaw = this.yawObj!;
		const pitch = this.pitchObj!;

		// Rotate yaw by dx
		yaw.rotation.y -= dx * sens;

		// Rotate pitch by dy, clamp to about +/- 89 deg
		pitch.rotation.x -= dy * sens;
		const limit = Math.PI / 2 - 0.01;
		pitch.rotation.x = Math.max(-limit, Math.min(limit, pitch.rotation.x));
	}
}

// ───────────────────────────── helpers ───────────────────────────────────────

function isTouchDevice(): boolean {
	if (typeof window === 'undefined') return false;
	return ('ontouchstart' in window) || navigator.maxTouchPoints > 0 || (navigator as any).msMaxTouchPoints > 0;
}

function clampVec2Magnitude(v: THREE.Vector2, maxLen: number): THREE.Vector2 {
	const l = v.length();
	if (l <= maxLen) return v.clone();
	return v.clone().multiplyScalar(maxLen / l);
}