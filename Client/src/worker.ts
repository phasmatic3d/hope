importScripts('/draco/draco_wasm_wrapper.js');

//import { decodePointCloud } from './dracoDecoder';

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