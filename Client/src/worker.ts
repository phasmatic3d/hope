/// <reference lib="webworker" />
// 1) Import the Emscripten glue
importScripts('/dracoCustom/draco_decoder.js');

const ModuleFactory = (self as any).DracoDecoderModule as (opts: any) => Promise<any>;
let Module: any;
const ModuleReady = ModuleFactory({
    locateFile: (f: string) => `/dracoCustom/${f}`
}).then((m) => {
  	Module = m;
});

self.onmessage = async (event) => {
	console.time("Draco");
	// 1) Wait for the module glue to finish initializing
	if (!Module) Module = await ModuleReady;

	// 2) Grab the incoming compressed bytes
	const raw = new Uint8Array(event.data.data);

	// 3) Allocate space in WASM heap
	const ptr = Module._malloc(raw.length);

	const heapBytes = Module.HEAPU8;
	heapBytes.set(raw, ptr);

	// 5) Decode
	const pcPtr = Module._decode_draco(ptr, raw.length);

	// 6) Free the input buffer
	Module._free(ptr);

	if (pcPtr === 0) {
		throw new Error("Draco decode failed");
	}

	// 7) Unpack the struct fields
	const positionsPtr = Module.getValue(pcPtr + 0, "i32");
	const colorsPtr    = Module.getValue(pcPtr + 4, "i32");
	const numPoints    = Module.getValue(pcPtr + 8, "i32");

	// 8) Slice out both arrays from the single HEAPU8 buffer
	const buffer = heapBytes.buffer;
	const positions = new Float32Array(buffer, positionsPtr, numPoints * 3);
	const colors    = new Uint8Array (buffer, colorsPtr,    numPoints * 3);

	// 9) Clean up the C struct
	Module._free_pointcloud(pcPtr);

	const positionsCopy = new Float32Array(positions);  // slice => new ArrayBuffer
	const colorsCopy    = new Uint8Array(colors);       // slice => new ArrayBuffer

	// 10) Post back, transferring ownership of the buffers
	self.postMessage(
		{ positions: positionsCopy, colors: colorsCopy, numPoints },
		[ positionsCopy.buffer, colorsCopy.buffer ]
	)

  	console.timeEnd("Draco")
};
/*
importScripts('/draco/draco_wasm_wrapper.js');


let decoderModule: any;

const initDecoder = async () => {
    // This loads the decoder in WASM or JS fallback.
    decoderModule = await (self as any).DracoDecoderModule({
        locateFile: (filename: string) => `/draco/${filename}`,
    });
};

initDecoder();

function decodePointCloud(decoderModule: any, rawBuffer: ArrayBuffer) {
    console.time("Draco")
    const buffer = new decoderModule.DecoderBuffer();
    buffer.Init(new Int8Array(rawBuffer), rawBuffer.byteLength);
  
    const decoder = new decoderModule.Decoder();
    const pointCloud = new decoderModule.PointCloud();
  
    const status = decoder.DecodeBufferToPointCloud(buffer, pointCloud);
    if (!status.ok() || pointCloud.ptr === 0) {
      throw new Error("Decoding failed: " + status.error_msg());
    }
    //Positions
    const numPoints = pointCloud.num_points();
    const attrId = decoder.GetAttributeId(pointCloud, decoderModule.POSITION);
    const posAttr = decoder.GetAttribute(pointCloud, attrId);
    const posData = new decoderModule.DracoFloat32Array();
    decoder.GetAttributeFloatForAllPoints(pointCloud, posAttr, posData);
    // Colors
    const colAttId = decoder.GetAttributeId(pointCloud, decoderModule.COLOR);
    const colAttr  = decoder.GetAttribute(pointCloud, colAttId);
    const colData  = new decoderModule.DracoUInt8Array();
    decoder.GetAttributeUInt8ForAllPoints(pointCloud, colAttr, colData);

  
    const positions = new Float32Array(numPoints * 3);
    const colors = new Uint8Array(numPoints * 3);
    for (let i = 0; i < posData.size(); i++) {
      positions[i] = posData.GetValue(i);
      colors[i] = colData.GetValue(i);
    }
  
    // Clean up Draco objects
    decoderModule.destroy(posData);
    decoderModule.destroy(posAttr);
    decoderModule.destroy(colData);
    decoderModule.destroy(colAttr);
    decoderModule.destroy(pointCloud);
    decoderModule.destroy(buffer);
    decoderModule.destroy(decoder);

    console.timeEnd("Draco")
    console.log(`Decoded ${numPoints} points`)
  
    return {positions, colors};
  }


interface IMPORT_DATA
{
    //decoderModule: any,
    data: ArrayBuffer
}

self.onmessage = (event: MessageEvent<IMPORT_DATA>) => {
    const data = event.data;
    const positions = decodePointCloud(decoderModule, data.data);
    //const positions = new Float32Array([1,2,3])
    postMessage(positions);
};
*/