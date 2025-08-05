import { time } from "three/tsl";

export function openConnection(
    //response : (data: ArrayBuffer) => void, 
    //response : (data: ArrayBuffer) => Promise<void>, 
    response: (data: ArrayBuffer) => Promise<{
        decodeTime: number;
        geometryUploadTime: number;
        frameTime: number;
        totalTime: number;
    }>,
    reject: (msg: string) => void
) {
    console.warn("WebSockets")
    // toggle this to switch between ws:// and wss://
    const USE_TLS = false;

    // host + port for each mode
    //const HOST = '192.168.1.135';
    const HOST = 'localhost';
    const PORT = 9002; // e.g. 9003 for TLS, 9002 for plain

    // pick the right protocol
    const protocol = USE_TLS ? 'wss' : 'ws';
    const url = `${protocol}://${HOST}:${PORT}`;
    // Connect to the WebSocket server
    console.warn(`Connecting over ${protocol.toUpperCase()} to ${url}`);

    const socket = new WebSocket(url);

    interface QueuedPacket {
        buf: ArrayBuffer;    // the point-cloud data
        round: number;      // snapshot of currentRound
        sendTS: number;     // snapshot of lastSendTimestamp
        receivedTS: number;
    }

    const packetQueue: QueuedPacket[] = [];
    let   lastSendTimestamp: number = -1;
    let   currentRound: number = -1;
    let   busy = false;    
  
    let syncInterval: ReturnType<typeof setInterval> | null = null;
    let pendingSync: { t0: number } | null = null;
    let lastOffset: number | null = null;


    // Set the binary type to 'arraybuffer'
    socket.binaryType = 'arraybuffer';

    function sendSyncRequest() {
        const t0 = Date.now();
        pendingSync = { t0 };
        socket.send(JSON.stringify({ type: "sync-request", t0 }));
    }
    
    function getCorrectedTime(): number {
        //return Date.now() + (lastOffset ?? 0);
        return Date.now();
    }

    async function processNextPacket(): Promise<void> {
        if (busy) return;
        busy = true;

        try{
            while (packetQueue.length) {
                const {buf, round, sendTS, receivedTS} = packetQueue.shift()!;
                if (round < 0 || sendTS < 0) continue;
                const poppedFromQueueAt = getCorrectedTime();
                const { decodeTime, geometryUploadTime, frameTime, totalTime } = await response(buf);
                const processedAt = getCorrectedTime();
                const message = JSON.stringify({
                    type:                       'ms-and-processing',
                    timestamp:                  receivedTS,
                    round:                      round,
                    pure_decode_ms:             decodeTime,
                    pure_geometry_upload_ms:    geometryUploadTime,
                    pure_render_ms:             frameTime,
                    pure_processing_ms:         totalTime,
                    wait_in_queue:              poppedFromQueueAt - receivedTS,
                    one_way_ms:                 receivedTS - sendTS,
                    one_way_plus_processing:    processedAt - sendTS
                })

                console.log("Sending message to server:\n" + message);

                socket.send(message);
            }
        }finally{
            busy = false;
            if (packetQueue.length) processNextPacket();
        }
    }

    // Event handler for when the connection opens
    socket.addEventListener('open', (event) => {
        console.warn('WebSocket connection opened.');
        //sendSyncRequest();
        //syncInterval = setInterval(sendSyncRequest, 10000);
    });

    // Event handler for when a message is received
    socket.addEventListener('message', async (event) => {
        if (typeof event.data === 'string') {
            let meta = JSON.parse(event.data);
            if (meta.type === 'broadcast-info') {
                currentRound = meta.round;
                lastSendTimestamp = meta.send_ts_ms;
                console.log(`Setting currentRound: ${currentRound}, setting lastSendTimestamp: ${lastSendTimestamp}`);
                return;
            }
        }

        if (event.data instanceof ArrayBuffer) {
            //packetQueue.push({
            //    buf: event.data,
            //    round: currentRound, 
            //    sendTS: lastSendTimestamp,
            //    receivedTS: getCorrectedTime()
            //});
            //if (!busy) processNextPacket();
            const receivedTS = getCorrectedTime();
            const poppedFromQueueAt = getCorrectedTime();
            const { decodeTime, geometryUploadTime, frameTime, totalTime } = await response(event.data);
            const processedAt = getCorrectedTime();
            const message = JSON.stringify({
                type:                       'ms-and-processing',
                timestamp:                  getCorrectedTime(),
                round:                      currentRound,
                pure_decode_ms:             decodeTime,
                pure_geometry_upload_ms:    geometryUploadTime,
                pure_render_ms:             frameTime,
                pure_processing_ms:         totalTime,
                wait_in_queue:              poppedFromQueueAt - receivedTS,
                one_way_ms:                 receivedTS - lastSendTimestamp,
                one_way_plus_processing:    processedAt - lastSendTimestamp
            })

            console.log("Sending message to server:\n" + message);

            socket.send(message);
        }   
    });

    // Error handling
    socket.addEventListener('error', (error) => {
        console.error('WebSocket error:', error);
        reject("Error");
    });

    socket.addEventListener('close', (event) => {
        if (syncInterval) clearInterval(syncInterval);
        syncInterval = null;
        console.warn('WebSocket connection closed.', event);
        reject("Close");
    });
}