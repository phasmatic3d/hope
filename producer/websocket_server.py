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
import socket
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from aiortc.sdp import candidate_from_sdp
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCConfiguration, RTCIceServer
import json
import datetime
import ipaddress
import shutil
import os

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def generate_self_signed_cert(path: Path):
    print("Generating self-signed certificate...")
    
    out_path = path / "cert"
    shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path)
    
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,)
 
    local_ip = get_local_ip()
    print(f"Adding IP to certificate: {local_ip}")

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"California"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"San Francisco"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"My Company"),
        x509.NameAttribute(NameOID.COMMON_NAME, local_ip),])

    builder = x509.CertificateBuilder()
    builder = builder.subject_name(subject)
    builder = builder.issuer_name(issuer)
    builder = builder.public_key(key.public_key())
    builder = builder.serial_number(x509.random_serial_number())
    builder = builder.not_valid_before(datetime.datetime.now(datetime.timezone.utc))
    
    builder = builder.not_valid_after(
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)
    )

    san_list = [
        x509.DNSName(u"localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv4Address(local_ip)) 
    ]
    
    builder = builder.add_extension(
        x509.SubjectAlternativeName(san_list),
        critical=False,)

    certificate = builder.sign(private_key=key, algorithm=hashes.SHA256())

    with open(out_path / "server.key", "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),))

    with open(out_path / "server.crt", "wb") as f:
        f.write(certificate.public_bytes(serialization.Encoding.PEM))
    
    print(f"Certificate and key saved to {out_path}")
    
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
        self.pcs: Dict[WebSocketServerProtocol, RTCPeerConnection] = {}
        
        # NEW: Store queues for each connected client
        self.client_queues: Dict[WebSocketServerProtocol, asyncio.Queue] = {}
        
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self._shutdown = False

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        # ... (Same initialization) ...
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
                    config = RTCConfiguration(iceServers=[])
                    pc = RTCPeerConnection(configuration=config)
                    self.pcs[websocket] = pc

                    @pc.on("datachannel")
                    def on_datachannel(channel):
                        print(f"✅ Data Channel Opened: {channel.label}")
                        
                        # --- NEW: Setup Queue and Worker ---
                        # Create a buffer of 50 packets (approx 1-2 seconds of video)
                        # If buffer fills, we will wait (or drop if we used put_nowait)
                        queue = asyncio.Queue(maxsize=50) 
                        self.client_queues[websocket] = queue
                        
                        # Start the background sender task for this specific client
                        asyncio.create_task(self._client_sender_task(websocket, channel, queue))

                    # ... (rest of WebRTC setup) ...
                    remote_desc = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                    await pc.setRemoteDescription(remote_desc)
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    response = {"type": pc.localDescription.type, "sdp": pc.localDescription.sdp}
                    await websocket.send(json.dumps(response))
                
                elif data["type"] == "ice":
                    if websocket in self.pcs and data.get("candidate"):
                        candidate_data = data["candidate"]
                        try:
                            candidate_str = candidate_data.get("candidate", "")
                            if not candidate_str: continue
                            candidate = candidate_from_sdp(candidate_str)
                            candidate.sdpMid = candidate_data.get("sdpMid")
                            candidate.sdpMLineIndex = candidate_data.get("sdpMLineIndex")
                            await self.pcs[websocket].addIceCandidate(candidate)
                        except Exception as e:
                            print(f"❌ ICE Error: {e}")
                    
        except Exception as e:
            print(f"❌ Signaling Error: {e}")
        finally:
            self.clients.discard(websocket)
            should_cleanup = True
            if websocket in self.pcs:
                pc = self.pcs[websocket]
                if pc.connectionState not in ["failed", "closed"]:
                    should_cleanup = False
            if should_cleanup:
                await self._cleanup_peer(websocket)

    async def _client_sender_task(self, ws, channel, queue):
        """Background task that manages the buffer for a single client."""
        try:
            while True:
                # 1. Get the next packet from the hidden buffer (Queue)
                packet = await queue.get()
                
                # 2. Backpressure Logic
                # Wait until the socket's internal buffer drains below 256KB.
                # This ensures we don't overwhelm the underlying SCTP connection.
                while channel.bufferedAmount > 256 * 1024:
                    await asyncio.sleep(0.005) # Sleep 5ms and check again
                
                # 3. Send (Room is available!)
                try:
                    if channel.readyState == "open":
                        channel.send(packet)
                    else:
                        break # Channel died
                except Exception:
                    break
                
                queue.task_done()
        except asyncio.CancelledError:
            pass
        finally:
            # Cleanup queue reference if task dies
            self.client_queues.pop(ws, None)

    async def _cleanup_peer(self, websocket):
        if websocket in self.client_queues:
            self.client_queues.pop(websocket)
            
        if websocket in self.pcs:
            pc = self.pcs.pop(websocket)
            try: await pc.close()
            except: pass

    def start(self):
        self.thread.start()

    def stop(self):
        self._shutdown = True
        # ... (keep existing stop logic) ...

    def broadcast(self, packet: bytes):
        """Push packet to all client buffers immediately"""
        if not self.client_queues:
            return

        def _push_to_queues():
            for ws, queue in list(self.client_queues.items()):
                try:
                    # 'put_nowait' will crash if queue is full (maxsize=50).
                    # Use 'await queue.put(packet)' if you want to BLOCK the python server 
                    # until room is made (Not recommended, slows down camera).
                    # Instead, we try to put, and if full, we drop (safest for live video).
                    
                    if not queue.full():
                        queue.put_nowait(packet)
                    else:
                        # Optional: Print warning
                        # print("Buffer full, dropping frame for client")
                        pass
                except Exception:
                    pass

        # We run this small loop in the asyncio thread
        self.loop.call_soon_threadsafe(_push_to_queues)

    # ... (Rest of class methods) ...
    def get_active_connections(self) -> int:
        return len(self.client_queues)
    
    def get_total_connections(self) -> int:
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