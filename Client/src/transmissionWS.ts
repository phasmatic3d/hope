// @ts-nocheck

import { time } from "three/tsl";
//import { socketHandlerResponse } from "./types";

export async function openConnection(
    response : (data: ArrayBuffer) => void,
    reject: (msg: string) => void
) {
    console.warn("WebSockets")
    // toggle this to switch between ws:// and wss://
    const USE_TLS = true;

    // host + port for each mode
    const HOST = window.location.hostname;//'192.168.1.152';
    //const HOST = 'localhost';
    const PORT = 9003;

    // pick the right protocol
    const protocol = USE_TLS ? 'wss' : 'ws';
    const url = `${protocol}://${HOST}:${PORT}`;
    // Connect to the WebSocket server
    console.warn(`Connecting over ${protocol.toUpperCase()} to ${url}`);

    const socket = new WebSocket(url);

    socket.binaryType = 'arraybuffer';

    // Event handler for when the connection opens
    socket.addEventListener('open', (event) => {
        console.warn('WebSocket connection opened.');
    });

     // Event handler for when a message is received
    socket.addEventListener('message', (event) => {
        //console.log("Received Message")
      if (event.data instanceof ArrayBuffer) {
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

    socket.addEventListener('close', (event) => {
        console.warn('WebSocket connection closed.', event);
        reject("Close");
    });
}