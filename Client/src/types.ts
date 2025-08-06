export type DecoderMessage = {
	positions: Float32Array,
	colors: Uint8Array, 
	numPoints: number,
	dracoDecodeTime: number
}

export type createPointCloudResult = {
	decodeTime: number, 
	geometryUploadTime: number,
	lastSceneUpdateTime: number,
}

export type socketHandlerResponse = {
	decodeTime: number,
	geometryUploadTime: number,
	frameTime: number,
	totalTime: number
}