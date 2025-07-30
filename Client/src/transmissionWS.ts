export function openConnetion(
    response : (data: ArrayBuffer) => void, 
    reject: (msg: string) => void
) {
  console.warn("WebSockets")
  // Connect to the WebSocket server
  const socket = new WebSocket('ws://localhost:9002');
  //const socket = new WebSocket('wss://192.168.1.155:9002');
  
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
    const receivedAt = Date.now();                      

    if (event.data instanceof ArrayBuffer) {
        //console.warn('Received ArrayBuffer:', event.data);
        socket.send
        (
            JSON.stringify
            (
                { 
                    type: 'received-timestamp', 
                    timestamp: receivedAt 
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