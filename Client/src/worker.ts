/// <reference lib="webworker" />

importScripts('/dracoCustom/draco_decoder.js');

import type { DecoderMessage } from './types';

let scratchPtr = 0;
const MAX_CHUNK_BYTES = 500_000;  

let Module: any = null;
const ModuleReady: Promise<void> = (self as any).DracoDecoderModule({
  locateFile: (file: string) => `/dracoCustom/${file}`
}).then((m: any) => {
  Module = m;
  scratchPtr = Module._malloc(MAX_CHUNK_BYTES);
});

let sharedView: Uint8Array;

self.onmessage = async (ev: MessageEvent<any>) => {
  const data = ev.data;

  // ─── Initialization: grab the SharedArrayBuffer ───
  if (data.type === 'init') {
    sharedView = new Uint8Array(data.sharedBuf);
    return;
  }

  // ─── Decode request ───
  if (data.type === 'decode') {
    if (!Module) await ModuleReady;
    const { offset, length } = data;
    const raw = sharedView.subarray(offset, offset + length);

    const dracoStart = performance.now();

    const ptr = Module._malloc(length);
    Module.HEAPU8.set(raw, ptr);

    const pcPtr = Module._decode_draco(ptr, length);
    Module._free(ptr);
    if (pcPtr === 0) {
      throw new Error('Draco decode failed');
    }

    const positionsPtr = Module.getValue(pcPtr + 0, 'i32');
    const colorsPtr    = Module.getValue(pcPtr + 4, 'i32');
    const numPoints    = Module.getValue(pcPtr + 8, 'i32');

    const heapBuf = Module.HEAPU8.buffer;
    const positionsView = new Float32Array(heapBuf, positionsPtr, numPoints * 3);
    const colorsView    = new Uint8Array (heapBuf, colorsPtr,    numPoints * 3);

    Module._free_pointcloud(pcPtr);

    // Copy out to new ArrayBuffers to transfer back
    const positionsCopy = new Float32Array(positionsView);
    const colorsCopy    = new Uint8Array(colorsView);
    const dracoDecodeTime = performance.now() - dracoStart;
    const msg: DecoderMessage = {
      positions: positionsCopy,
      colors:    colorsCopy,
      numPoints,
      dracoDecodeTime
    };
    self.postMessage(msg, [positionsCopy.buffer, colorsCopy.buffer]);
	
<<<<<<< HEAD
	const { offset, length } = data;
	const raw = sharedEncodedView.subarray(offset, offset + length);

	const dracoStart = performance.now();

	//const ptr = Module._malloc(length);
	Module.HEAPU8.set(raw, scratchPtr);
	const pcPtr = Module._decode_draco(scratchPtr, length);
	//Module._free(ptr);
	if (pcPtr === 0) throw new Error('Draco decode failed');

	// unpack
	const positionsPtr = Module.getValue(pcPtr + 0, 'i32');
	const colorsPtr    = Module.getValue(pcPtr + 4, 'i32');
	const numPoints    = Module.getValue(pcPtr + 8, 'i32');
	const heap = Module.HEAPU8.buffer;
	const inPos = new Float32Array(heap, positionsPtr, numPoints * 3);
	const inCol = new Uint8Array (heap, colorsPtr,    numPoints * 3);

	Module._free_pointcloud(pcPtr);

	decodedPosView.set(inPos, offset * 3);
	decodedColView.set(inCol, offset * 3);

	const dracoDecodeTime = performance.now() - dracoStart;
	const msg: DecoderMessage = {
		numPoints,
		dracoDecodeTime
	};
	self.postMessage( msg );
	}
};
=======
  }
};
>>>>>>> parent of e60cb37 (successfully writing into shared buffer)
