/// <reference lib="webworker" />

importScripts('/dracoCustom/draco_decoder.js');

import type { DecoderMessage } from './types';

let Module: any = null;
const ModuleReady: Promise<void> = (self as any).DracoDecoderModule({
  locateFile: (file: string) => `/dracoCustom/${file}`
}).then((m: any) => {
  Module = m;
});

let sharedEncodedView: Uint8Array;
let decodedPosView: Float32Array;
let decodedColView: Uint8Array;


self.onmessage = async (ev: MessageEvent<any>) => {
   const data = ev.data;

  // Initialization request
	if (data.type === 'init') {
		sharedEncodedView = new Uint8Array(data.sharedEncodedBuffer);
		decodedPosView    = new Float32Array(data.decodedPosBuffer);
		decodedColView    = new Uint8Array (data.decodedColBuffer);
		return;
	}

  // Decode request
if (data.type === 'decode') {
	if (!Module) await ModuleReady;
	
	const { offset, length, pointOffset } = data;
	const raw = sharedEncodedView.subarray(offset, offset + length);

	const dracoStart = performance.now();

	const ptr = Module._malloc(length);
	Module.HEAPU8.set(raw, ptr);
	const pcPtr = Module._decode_draco(ptr, length);
	Module._free(ptr);
	if (pcPtr === 0) throw new Error('Draco decode failed');

	// unpack
	const positionsPtr = Module.getValue(pcPtr + 0, 'i32');
	const colorsPtr    = Module.getValue(pcPtr + 4, 'i32');
	const numPoints    = Module.getValue(pcPtr + 8, 'i32');
	const heap = Module.HEAPU8.buffer;
	const inPos = new Float32Array(heap, positionsPtr, numPoints * 3);
	const inCol = new Uint8Array (heap, colorsPtr,    numPoints * 3);

	Module._free_pointcloud(pcPtr);

	decodedPosView.set(inPos, pointOffset * 3);
	decodedColView.set(inCol, pointOffset * 3);

	const dracoDecodeTime = performance.now() - dracoStart;
	const msg: DecoderMessage = {
		numPoints,
		dracoDecodeTime
	};
	self.postMessage( msg );
	}
};
