export type DecoderMessage = {
	numPoints: number,
	dracoDecodeTime: number
}

export type createPointCloudResult = {
	decodeTime: number, 
	geometryUploadTime: number,
	lastSceneUpdateTime: number,
	chunkDecodeTimes: number[],
}

export type socketHandlerResponse = {
	decodeTime: number,
	geometryUploadTime: number,
	frameTime: number,
	totalTime: number,
	chunkDecodeTimes: number[]
}