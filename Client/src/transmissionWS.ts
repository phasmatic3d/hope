export function openConnetion(
    response : (data: ArrayBuffer) => void, 
    reject: (msg: string) => void
) {
    console.warn("WebSockets")
    // toggle this to switch between ws:// and wss://
    const USE_TLS = true;

    // host + port for each mode
    const HOST = 'localhost';
    const PORT = USE_TLS ? 9003 : 9002; // e.g. 9003 for TLS, 9002 for plain

    // pick the right protocol
    const protocol = USE_TLS ? 'wss' : 'ws';
    const url = `${protocol}://${HOST}:${PORT}`;
    // Connect to the WebSocket server
    console.warn(`Connecting over ${protocol.toUpperCase()} to ${url}`);

    const socket = new WebSocket(url);
  
    let currentRound: number | null = null;

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
        }
        return;
    }

    if (event.data instanceof ArrayBuffer) {
        //console.warn('Received ArrayBuffer:', event.data);
        const receivedAt = Date.now();

        // send back timestamp *and* the round
        socket.send(
            JSON.stringify
            (
                {
                    type:      'received-timestamp',
                    timestamp: receivedAt,
                    round:     currentRound
                }
            )
        );

        response(event.data);

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