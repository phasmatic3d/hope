export function openConnection(
    response : (data: ArrayBuffer) => void, 
    reject: (msg: string) => void
) {
    console.warn("WebSockets")
    // toggle this to switch between ws:// and wss://
    const USE_TLS = true;

    // host + port for each mode
    const HOST = '192.168.1.135';
    const PORT = 9003; // e.g. 9003 for TLS, 9002 for plain

    // pick the right protocol
    const protocol = USE_TLS ? 'wss' : 'ws';
    const url = `${protocol}://${HOST}:${PORT}`;
    // Connect to the WebSocket server
    console.warn(`Connecting over ${protocol.toUpperCase()} to ${url}`);

    const socket = new WebSocket(url);
  
    let currentRound: number | null = null;
    let lastSendTimestamp: number | null = null;

    // Set the binary type to 'arraybuffer'
    socket.binaryType = 'arraybuffer';

    // Event handler for when the connection opens
    socket.addEventListener('open', (event) => {
        console.warn('WebSocket connection opened.');
        // Optionally, you can send a message to the server here
        socket.send('Hello Server!');
    });

    // Event handler for when a message is received
    socket.addEventListener('message', (event) => {
      //console.warn("Receive Message")
        if (typeof event.data === 'string') {
            // this is our JSON info frame
            let meta = JSON.parse(event.data);
            if (meta.type === 'broadcast-info') {
                currentRound = meta.round;
                lastSendTimestamp = meta.send_ts_ms;
            }
            return;
        }

        if (event.data instanceof ArrayBuffer) {
            //console.warn('Received ArrayBuffer:', event.data);
            const receivedAt = Date.now();

            response(event.data);

            const timeAfterProcessing = Date.now();

            if(lastSendTimestamp !== null){
                const one_way_ms              = receivedAt - lastSendTimestamp;
                const one_way_plus_processing = timeAfterProcessing - lastSendTimestamp;

                // send back timestamp *and* the round
                socket.send(
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

        } else {
            console.warn('Received non-ArrayBuffer message:', event.data);
        }
  });

  // Error handling
  socket.addEventListener('error', (error) => {
      console.error('WebSocket error:', error);
      reject("Error");
  });

  // When the connection is closed
  socket.addEventListener('close', (event) => {
      console.warn('WebSocket connection closed.', event);
      reject("Close");
  });
}