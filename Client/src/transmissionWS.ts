export function openConnection(
    response : (data: ArrayBuffer) => void, 
    reject: (msg: string) => void
) {
    console.warn("WebSockets")
    // toggle this to switch between ws:// and wss://
    const USE_TLS = true;

    // host + port for each mode
    //const HOST = '192.168.1.135';
    const HOST = 'localhost';
    const PORT = 9003; // e.g. 9003 for TLS, 9002 for plain

    // pick the right protocol
    const protocol = USE_TLS ? 'wss' : 'ws';
    const url = `${protocol}://${HOST}:${PORT}`;
    // Connect to the WebSocket server
    console.warn(`Connecting over ${protocol.toUpperCase()} to ${url}`);

    const socket = new WebSocket(url);
  
    let currentRound: number | null = null;
    let lastSendTimestamp: number | null = null;
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
        return Date.now() + (lastOffset ?? 0);
    }

    // Event handler for when the connection opens
    socket.addEventListener('open', (event) => {
        console.warn('WebSocket connection opened.');
        sendSyncRequest();
        syncInterval = setInterval(sendSyncRequest, 10000);
    });

    // Event handler for when a message is received
    socket.addEventListener('message', (event) => {
        //console.warn("Receive Message")
        if (typeof event.data === 'string') {
            let meta = JSON.parse(event.data);

            if (meta.type === 'broadcast-info') {
                currentRound = meta.round;
                lastSendTimestamp = meta.send_ts_ms;
                return;
            }

            if (meta.type === 'sync-response' && pendingSync && meta.t0 === pendingSync.t0) {
                const t1 = Date.now();
                const serverTime = meta.server_time;
                const rtt = t1 - pendingSync.t0;
                const estimatedOffset = serverTime + rtt / 2 - t1;
                lastOffset = estimatedOffset;

                console.warn(`[SYNC] RTT: ${rtt}ms, Offset: ${estimatedOffset}ms`);
                // Clear pending sync
                pendingSync = null;
                return;
            }
        }

        if (event.data instanceof ArrayBuffer) {
            //console.warn('Received ArrayBuffer:', event.data);
            const receivedAt = getCorrectedTime();

            response(event.data);
            response(event.data);

            const timeAfterProcessing = getCorrectedTime();

            if(lastSendTimestamp !== null) {
                const one_way_ms              = receivedAt - lastSendTimestamp;
                const one_way_plus_processing = timeAfterProcessing - lastSendTimestamp;

                // send back timestamp *and* the round
                socket.send
                (
                    JSON.stringify
                    (
                        {
                            type:      'ms-and-processing',
                            round:      currentRound,
                            one_way_ms: one_way_ms,
                            one_way_plus_processing: one_way_plus_processing
                        }
                    )
                );
            }

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