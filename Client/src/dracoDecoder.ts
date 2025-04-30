/*export async function initDracoDecoder() {
    const decoderModule = await (window as any).DracoDecoderModule({
      locateFile: (filename: string) => `/draco/${filename}`,
    });
    return decoderModule;
}*/

export async function initDracoDecoder() {
    return new Promise<any>((resolve, reject) => {
      // Load Draco WASM wrapper script dynamically
      const script = document.createElement('script');
      script.src = '/draco/draco_wasm_wrapper.js';  // Path to your draco_wasm_wrapper.js
      script.onload = () => {
        // Now Draco WASM module is available
        const decoderModule = (window as any).DracoDecoderModule({
          locateFile: (filename: string) => `/draco/${filename}`,
        });
  
        // Resolve the promise once Draco is initialized
        decoderModule.then(resolve).catch(reject);
      };
      script.onerror = reject;
      document.body.appendChild(script);
    });
}

export function decodePointCloud(decoderModule: any, rawBuffer: ArrayBuffer) {
    console.time("Draco")
    const buffer = new decoderModule.DecoderBuffer();
    buffer.Init(new Int8Array(rawBuffer), rawBuffer.byteLength);
  
    const decoder = new decoderModule.Decoder();
    const pointCloud = new decoderModule.PointCloud();
  
    const status = decoder.DecodeBufferToPointCloud(buffer, pointCloud);
    if (!status.ok() || pointCloud.ptr === 0) {
      throw new Error("Decoding failed: " + status.error_msg());
    }
  
    const numPoints = pointCloud.num_points();
    const attrId = decoder.GetAttributeId(pointCloud, decoderModule.POSITION);
    const posAttr = decoder.GetAttribute(pointCloud, attrId);
    const posData = new decoderModule.DracoFloat32Array();
  
    decoder.GetAttributeFloatForAllPoints(pointCloud, posAttr, posData);
  
    const positions = new Float32Array(numPoints * 3);
    for (let i = 0; i < posData.size(); i++) {
      positions[i] = posData.GetValue(i);
    }
  
    // Clean up Draco objects
    decoderModule.destroy(posData);
    decoderModule.destroy(posAttr);
    decoderModule.destroy(pointCloud);
    decoderModule.destroy(buffer);
    decoderModule.destroy(decoder);

    console.timeEnd("Draco")
    console.log(`Decoded ${numPoints} points`)
  
    return positions;
  }