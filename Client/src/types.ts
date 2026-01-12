export type DecoderMessage = {
        type: 'decoded',
        numPoints: number,
        frameId: number,
        decodeTimeMs: number
}
