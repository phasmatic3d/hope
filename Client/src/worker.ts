/// <reference lib="webworker" />

importScripts('/dracoCustom/draco_decoder.js');

import type { DecoderMessage } from './types';

let scratchPtr = 0;
const MAX_CHUNK_BYTES = 15_500_000;  

let Module: any = null;
const ModuleReady: Promise<void> = (self as any).DracoDecoderModule({
  locateFile: (file: string) => `/dracoCustom/${file}`
}).then((m: any) => {
  Module = m;
  scratchPtr = Module._malloc(MAX_CHUNK_BYTES);
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
		const { offset, length, writeIndex } = data;
		const raw = sharedEncodedView.subarray(offset, offset + length);

		Module.HEAPU8.set(raw, scratchPtr);
		const pcPtr = Module._decode_draco(scratchPtr, length);
		if (pcPtr === 0) throw new Error('Draco decode failed');

		const positionsPtr = Module.getValue(pcPtr + 0, 'i32');
		const colorsPtr    = Module.getValue(pcPtr + 4, 'i32');
		const numPoints    = Module.getValue(pcPtr + 8, 'i32');

		const heap  = Module.HEAPU8.buffer;
		const inPos = new Float32Array(heap, positionsPtr, numPoints * 3);
		const inCol = new Uint8Array(heap, colorsPtr,    numPoints * 3);

		decodedPosView.set(inPos, writeIndex * 3);
		decodedColView.set(inCol, writeIndex * 3);

		self.postMessage({ type: 'decoded', numPoints } as DecoderMessage);
	}
};
