export function openConnection(
    response: (data: ArrayBuffer) => void,
    reject: (msg: string) => void
) {
    console.warn("🚀 Initializing WebRTC Connection...");

    const SIGNALING_PORT = 9003; 
    const HOST = '192.168.1.152';
    const signalingUrl = `wss://${HOST}:${SIGNALING_PORT}`;

    let ws: WebSocket | null = null;
    let pc: RTCPeerConnection | null = null;
    let dc: RTCDataChannel | null = null;
    let isConnected = false;
    let wsCloseTimeout: number | null = null;

    // ICE candidate queue for candidates received before remote description
    const iceCandidateQueue: RTCIceCandidateInit[] = [];
    let remoteDescriptionSet = false;

    async function setupConnection() {
        try {
            ws = new WebSocket(signalingUrl);
            
            // Create peer connection with STUN server
            pc = new RTCPeerConnection({
                iceServers: [
                    { urls: 'stun:stun.l.google.com:19302' },
                    { urls: 'stun:stun1.l.google.com:19302' }
                ]
            });

            // Create data channel BEFORE creating offer
            dc = pc.createDataChannel("pointcloud", { 
                ordered: false, 
                maxRetransmits: 0 
            });

            dc.binaryType = "arraybuffer";

            dc.onopen = () => {
                console.warn("✅ WebRTC Data Channel OPEN!");
                isConnected = true;
                
                // Close WebSocket after data channel is established
                wsCloseTimeout = window.setTimeout(() => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        console.log("🔌 Closing signaling WebSocket");
                        ws.close();
                    }
                }, 3000);
            };

            dc.onmessage = (event) => {
                if (event.data instanceof ArrayBuffer) {
                    response(event.data);
                }
            };

            dc.onerror = (error) => {
                console.error("❌ Data Channel Error:", error);
                if (!isConnected) {
                    reject("Data channel error");
                }
            };

            dc.onclose = () => {
                console.warn("❌ Data Channel CLOSED");
                if (!isConnected) {
                    reject("Data channel closed before connection");
                }
            };

            // ICE candidate handler
            pc.onicecandidate = (event) => {
                if (event.candidate && ws && ws.readyState === WebSocket.OPEN) {
                    console.log("🧊 Sending ICE candidate to server");
                    ws.send(JSON.stringify({
                        type: 'ice',
                        candidate: {
                            candidate: event.candidate.candidate,
                            sdpMid: event.candidate.sdpMid,
                            sdpMLineIndex: event.candidate.sdpMLineIndex
                        }
                    }));
                }
            };

            // Connection state monitoring
            pc.onconnectionstatechange = () => {
                console.log("🔌 Connection State:", pc!.connectionState);
                
                if (pc!.connectionState === "connected") {
                    console.log("✅ WebRTC Peer Connection ESTABLISHED");
                }
                
                if (pc!.connectionState === "failed") {
                    console.error("❌ Connection failed");
                    reject(`WebRTC connection failed`);
                    cleanup();
                }
                
                if (pc!.connectionState === "disconnected") {
                    console.warn("⚠️ Connection disconnected");
                }
            };

            pc.oniceconnectionstatechange = () => {
                console.log("🧊 ICE Connection State:", pc!.iceConnectionState);
                
                if (pc!.iceConnectionState === "failed") {
                    console.error("❌ ICE connection failed");
                    reject("ICE connection failed");
                }
            };

            // WebSocket handlers
            ws.onopen = async () => {
                console.log("🔌 Signaling Connected. Creating Offer...");
                
                try {
                    const offer = await pc!.createOffer();
                    await pc!.setLocalDescription(offer);

                    ws!.send(JSON.stringify({
                        type: offer.type,
                        sdp: offer.sdp
                    }));
                    
                    console.log("📤 Offer sent to server");
                } catch (error) {
                    console.error("❌ Failed to create offer:", error);
                    reject("Failed to create WebRTC offer");
                }
            };

            ws.onmessage = async (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'answer') {
                        console.log("📥 Received Answer. Setting remote description...");
                        const answer = new RTCSessionDescription({
                            type: data.type,
                            sdp: data.sdp
                        });
                        
                        await pc!.setRemoteDescription(answer);
                        remoteDescriptionSet = true;
                        
                        // Process queued ICE candidates
                        console.log(`🧊 Processing ${iceCandidateQueue.length} queued ICE candidates`);
                        for (const candidate of iceCandidateQueue) {
                            await pc!.addIceCandidate(new RTCIceCandidate(candidate));
                        }
                        iceCandidateQueue.length = 0;
                    } 
                    else if (data.type === 'ice' && data.candidate) {
                        console.log("🧊 Received ICE candidate from server");
                        
                        const candidate = new RTCIceCandidate(data.candidate);
                        
                        // Queue candidates if remote description not set yet
                        if (!remoteDescriptionSet) {
                            console.log("🧊 Queueing ICE candidate (remote description not set)");
                            iceCandidateQueue.push(data.candidate);
                        } else {
                            await pc!.addIceCandidate(candidate);
                        }
                    }
                } catch (error) {
                    console.error("❌ Error handling signaling message:", error);
                    reject("Signaling message handling failed");
                }
            };
            
            ws.onerror = (e) => {
                console.error("❌ WebSocket Error:", e);
                reject(`Signaling Error`);
            };

            ws.onclose = (e) => {
                console.log("🔌 WebSocket closed:", e.code, e.reason);
            };

        } catch (error) {
            console.error("❌ Setup error:", error);
            reject(`Setup failed: ${error}`);
            cleanup();
        }
    }

    function cleanup() {
        if (wsCloseTimeout) {
            clearTimeout(wsCloseTimeout);
        }
        
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
        
        if (dc && dc.readyState === "open") {
            dc.close();
        }
        
        if (pc && pc.connectionState !== "closed") {
            pc.close();
        }
        
        ws = null;
        pc = null;
        dc = null;
    }

    // Start the connection
    setupConnection();

    // Return cleanup function
    return cleanup;
}