import asyncio
import ssl
import threading
import websockets
from websockets.server import serve
from websockets import WebSocketServerProtocol
from typing import Set, Dict
from pathlib import Path
import http.server
import socketserver
import functools

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCConfiguration, RTCIceServer
import json

class SecureWebSocketServer:
    def __init__(self, host: str, port: int, cert_folder: str, cert_file: str, key_file: str):
        self.host = host
        self.port = port
        self.cert_folder = Path(cert_folder)
        self.cert_file = cert_file
        self.key_file = key_file
        
        self.clients: Set[WebSocketServerProtocol] = set()
        self._client_lock = asyncio.Lock()  # For thread-safe client management
 
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self._shutdown_event = asyncio.Event()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)

        if not self.cert_folder.exists():
            print(f"Error: Cert folder {self.cert_folder} not found.")
            return

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            ssl_context.load_cert_chain(
                self.cert_folder / self.cert_file,
                self.cert_folder / self.key_file
            )
        except Exception as e:
            print(f"SSL Error: {e}")
            return

        print(f"WSS Server listening on wss://{self.host}:{self.port}")
        
        async def run_server():
            async with serve(
                self._handler, 
                self.host, 
                self.port, 
                ssl=ssl_context, 
                max_size=None,
                compression=None  # Disable compression for better performance
            ):
                await self._shutdown_event.wait()
        
        self.loop.run_until_complete(run_server())

    async def _handler(self, websocket: WebSocketServerProtocol):
        """Handle individual WebSocket connections"""
        # Add client
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        try:
            # Keep connection alive and optionally handle incoming messages
            async for message in websocket:
                # If you need to handle incoming messages, process them here
                # For now, just ignore them as this is primarily for broadcasting
                pass
                
        except Exception as e:
            print(f"Client handler error: {e}")
        finally:
            # Remove client
            self.clients.discard(websocket)  # discard() won't raise if not present
            print(f"Client disconnected. Total clients: {len(self.clients)}")

    def start(self):
        """Start the WebSocket server thread"""
        self.thread.start()
        print("WebSocket server thread started")

    def stop(self):
        """Stop the server gracefully"""
        print("Stopping WebSocket server...")
        
        async def _stop():
            self._shutdown_event.set()
            
            # Close all client connections
            if self.clients:
                await asyncio.gather(
                    *[client.close() for client in list(self.clients)],
                    return_exceptions=True
                )
            self.clients.clear()
        
        if self.loop and self.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_stop(), self.loop)
            try:
                future.result(timeout=5)
            except Exception as e:
                print(f"Error during shutdown: {e}")
        
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def broadcast(self, packet: bytes):
        """Broadcast binary data to all connected clients"""
        if not self.clients:
            return

        async def _broadcast():
            # Make a copy to avoid modification during iteration
            active_clients = list(self.clients)
            
            if not active_clients:
                return
            
            # Send to all clients concurrently
            results = await asyncio.gather(
                *[self._send_to_client(ws, packet) for ws in active_clients],
                return_exceptions=True
            )
            
            # Check for failures
            failed = sum(1 for r in results if isinstance(r, Exception))
            if failed > 0:
                print(f"Broadcast: {failed}/{len(active_clients)} clients failed")

        try:
            future = asyncio.run_coroutine_threadsafe(_broadcast(), self.loop)
            # Don't wait for completion to avoid blocking the calling thread
            # But you can check later: future.result(timeout=0.1)
        except Exception as e:
            print(f"Broadcast error: {e}")

    async def _send_to_client(self, ws: WebSocketServerProtocol, packet: bytes):
        """Send packet to a single client with error handling"""
        try:
            await ws.send(packet)
        except Exception as e:
            print(f"Send error to client: {e}")
            self.clients.discard(ws)
            raise  # Re-raise to be caught by gather()

    def get_client_count(self) -> int:
        """Get the current number of connected clients"""
        return len(self.clients)

    def is_running(self) -> bool:
        """Check if the server thread is running"""
        return self.thread.is_alive()

class SecureWebRTCServer:
    def __init__(self, host: str, port: int, cert_folder: str, cert_file: str, key_file: str):
        self.host = host
        self.port = port
        self.cert_folder = Path(cert_folder)
        self.cert_file = cert_file
        self.key_file = key_file
        
        self.clients: Set[WebSocketServerProtocol] = set()
        self.data_channels: Dict[WebSocketServerProtocol, any] = {}
        self.pcs: Dict[WebSocketServerProtocol, RTCPeerConnection] = {}
        
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self._shutdown = False

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)

        if not self.cert_folder.exists():
            print(f"Error: Cert folder {self.cert_folder} not found.")
            return

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            ssl_context.load_cert_chain(
                self.cert_folder / self.cert_file,
                self.cert_folder / self.key_file
            )
        except Exception as e:
            print(f"SSL Error: {e}")
            return

        print(f"Signaling Server (WSS) listening on wss://{self.host}:{self.port}")
        
        async def run_server():
            async with serve(self._handler, self.host, self.port, ssl=ssl_context):
                await asyncio.Future()  # Run forever
        
        self.loop.run_until_complete(run_server())

    async def _handler(self, websocket: WebSocketServerProtocol):
        self.clients.add(websocket)
        pc = None
        
        print(f"Client connected via WebSocket")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data["type"] == "offer":
                    print("📥 Received WebRTC Offer from client")
                    
                    # Create RTCConfiguration with RTCIceServer objects
                    ice_servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
                    config = RTCConfiguration(iceServers=ice_servers)
                    
                    # Create peer connection with proper configuration
                    pc = RTCPeerConnection(configuration=config)
                    self.pcs[websocket] = pc

                    # Handle ICE candidates from server side
                    @pc.on("icecandidate")
                    async def on_icecandidate(candidate):
                        if candidate:
                            print(f"🧊 Sending ICE candidate to client")
                            await websocket.send(json.dumps({
                                "type": "ice",
                                "candidate": {
                                    "candidate": candidate.candidate,
                                    "sdpMid": candidate.sdpMid,
                                    "sdpMLineIndex": candidate.sdpMLineIndex
                                }
                            }))

                    @pc.on("datachannel")
                    def on_datachannel(channel):
                        print(f"✅ Data Channel Opened: {channel.label}")
                        self.data_channels[websocket] = channel

                        @channel.on("close")
                        def on_close():
                            print(f"❌ Data Channel Closed: {channel.label}")
                            self.data_channels.pop(websocket, None)
                        
                        @channel.on("error")
                        def on_error(error):
                            print(f"❌ Data Channel Error: {error}")

                    @pc.on("connectionstatechange")
                    async def on_connectionstatechange():
                        print(f"🔌 Connection state: {pc.connectionState}")
                        if pc.connectionState in ["failed", "closed"]:
                            await self._cleanup_peer(websocket)

                    @pc.on("iceconnectionstatechange")
                    async def on_iceconnectionstatechange():
                        print(f"🧊 ICE Connection state: {pc.iceConnectionState}")
                        if pc.iceConnectionState == "failed":
                            print("❌ ICE connection failed!")

                    @pc.on("icegatheringstatechange")
                    async def on_icegatheringstatechange():
                        print(f"🧊 ICE Gathering state: {pc.iceGatheringState}")

                    # Set remote description
                    remote_desc = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                    await pc.setRemoteDescription(remote_desc)
                    print("✅ Remote description set")
                    
                    # Create and set local description
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    print("✅ Local description set")

                    # Send answer back
                    response = {
                        "type": pc.localDescription.type,
                        "sdp": pc.localDescription.sdp
                    }
                    await websocket.send(json.dumps(response))
                    print("📤 Sent WebRTC Answer to client")
                
                elif data["type"] == "ice":
                    # Handle ICE candidates from client
                    if websocket in self.pcs and data.get("candidate"):
                        print(f"🧊 Received ICE candidate from client")
                        candidate_data = data["candidate"]
                        
                        try:
                            # Create RTCIceCandidate
                            candidate = RTCIceCandidate(
                                candidate=candidate_data.get("candidate", ""),
                                sdpMid=candidate_data.get("sdpMid"),
                                sdpMLineIndex=candidate_data.get("sdpMLineIndex")
                            )
                            
                            await self.pcs[websocket].addIceCandidate(candidate)
                            print("✅ ICE candidate added")
                        except Exception as e:
                            print(f"❌ Failed to add ICE candidate: {e}")
                    
        except Exception as e:
            print(f"❌ Signaling Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.clients.discard(websocket)
            await self._cleanup_peer(websocket)

    async def _cleanup_peer(self, websocket: WebSocketServerProtocol):
        """Clean up peer connection and data channel"""
        print(f"🧹 Cleaning up peer for websocket")
        
        if websocket in self.data_channels:
            channel = self.data_channels.pop(websocket)
            try:
                if channel.readyState != "closed":
                    channel.close()
            except:
                pass
        
        if websocket in self.pcs:
            pc = self.pcs.pop(websocket)
            try:
                if pc.connectionState != "closed":
                    await pc.close()
                    print("✅ Peer connection closed")
            except Exception as e:
                print(f"⚠️ Error closing peer connection: {e}")

    def start(self):
        """Start the server thread"""
        self.thread.start()
        print("✅ WebRTC Server thread started")

    def stop(self):
        """Stop the server gracefully"""
        self._shutdown = True
        
        async def _stop():
            # Close all peer connections
            for websocket, pc in list(self.pcs.items()):
                await self._cleanup_peer(websocket)
        
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(_stop(), self.loop)
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        if self.thread.is_alive():
            self.thread.join(timeout=5)

    def broadcast(self, packet: bytes):
        """Broadcast data to all connected clients"""
        if not self.data_channels:
            return

        async def _broadcast():
            disconnected = []
            
            for websocket, channel in list(self.data_channels.items()):
                try:
                    if channel.readyState == "open":
                        if channel.bufferedAmount < 150000:
                            channel.send(packet)
                        else:
                            print(f"⚠️ Buffer full for channel: {channel.bufferedAmount} bytes")
                    else:
                        print(f"⚠️ Channel not open: {channel.readyState}")
                        disconnected.append(websocket)
                except Exception as e:
                    print(f"❌ WebRTC Send Error: {e}")
                    disconnected.append(websocket)
            
            # Clean up disconnected channels
            for ws in disconnected:
                self.data_channels.pop(ws, None)

        try:
            asyncio.run_coroutine_threadsafe(_broadcast(), self.loop)
        except Exception as e:
            print(f"❌ Broadcast error: {e}")

    def get_active_connections(self) -> int:
        """Get number of active data channels"""
        return len([ch for ch in self.data_channels.values() if ch.readyState == "open"])
    
    def get_total_connections(self) -> int:
        """Get total number of peer connections"""
        return len(self.pcs)
         
class SecureHTTPThreadedServer:
    def __init__(self, host: str, port: int, cert_folder: str, cert_file: str, key_file: str, directory: str):
        self.host = host
        self.port = port
        self.cert_folder = Path(cert_folder)
        self.cert_file = cert_file
        self.key_file = key_file
        self.directory = directory
        self.thread = threading.Thread(target=self._run_server, daemon=True)

    def _run_server(self):
        """Runs the HTTPS server with Cross-Origin Isolation headers."""
        
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            ssl_context.load_cert_chain(
                self.cert_folder / self.cert_file,
                self.cert_folder / self.key_file
            )
        except Exception as e:
            print(f"HTTP SSL Error: {e}")
            return

        directory_to_serve = self.directory

        class COOPCOEPHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory_to_serve, **kwargs)

            def end_headers(self):
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
                super().end_headers()
        try:
            with socketserver.TCPServer((self.host, self.port), COOPCOEPHandler) as httpd:
                httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)
                print(f"HTTPS Clienting serving '{self.directory}' at https://{self.host}:{self.port}")
                httpd.serve_forever()
        except OSError as e:
            print(f"HTTP Port {self.port} likely in use. Error: {e}")

    def start(self):
        self.thread.start()