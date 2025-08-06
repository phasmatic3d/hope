# 🛠️ Local TLS Setup for Your WebSocket Server

All commands assume you’re in your project root:

```
~/Github/hope/producer/src/broadcaster_wrapper
```

---

## 1. Create a “cert” workspace

```bash
mkdir -p cert && cd cert
```

# Generate your own Certificate Authority (CA)

## 2a. CA private key
```
openssl genrsa -out myCA.key 2048
```

## 2b. CA self-signed root certificate (10 years)
```
openssl req -x509 -new -nodes \
  -key myCA.key \
  -sha256 -days 3650 \
  -out myCA.pem \
  -subj "/C=US/ST=State/L=City/O=Local Dev CA/CN=Local Dev Root CA"
```

# Create a server key & CSR

## 3a. Server private key
```
openssl genrsa -out server.key 2048
```

## 3b. Certificate Signing Request (CSR)
```
openssl req -new \
  -key server.key \
  -out server.csr \
  -subj "/C=US/ST=State/L=City/O=Hope Dev/CN=Hope Producer"
```

# Create the SAN extension file
Create a file named `server.ext` next to your CSR with these contents:


```
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = "The Server's local IP goes here"
IP.1  = 127.0.0.1
IP.2  = ::1
IP.3  = "The Server's local IP goes here"
```

# Sign the CSR with your CA, injecting the SANs

```
openssl x509 -req \
  -in server.csr \
  -CA myCA.pem -CAkey myCA.key -CAcreateserial \
  -out server.crt \
  -days 365 -sha256 \
  -extfile server.ext
```

# Install your CA into the trust store 
**For Linux/Ubuntu**
```
# Copy the CA cert into the system store
sudo cp myCA.pem /usr/local/share/ca-certificates/myCA.crt

# Update the CA bundle
sudo update-ca-certificates
```

**For Windows**
```
openssl x509 -outform DER -in myCA.pem -out myCA.cer

certutil -addstore Root .\myCA.cer
```
# Test if the key is trusted by the system

```
export NODE_EXTRA_CA_CERTS="/usr/local/share/ca-certificates/myCA.crt"
```

# Make the browser trust the key 
**For Linux/Ubuntu**
```
sudo apt install libnss3-tools
```


```
certutil -d sql:"$HOME/.pki/nssdb" \
        -A -t "C,," \
        -n "Local Dev Root CA" \
        -i ~/Github/hope/producer/src/broadcaster_wrapper/cert/myCA.pem
```

**For Windows**
```bash
#on the working directory copied you copied your myCA.crt on the viewer machine
certutil -addstore Root .\myCA.crt 
```

# Build the Client in the producer's source
1. **Build the client** (on the producer machine):
   ```bash
   cd /path/to/client/root
   npm install        # if npm isn't available
   npm run build      # outputs to dist/ or build/ folder
   ```

2. **Copy the built files into the server’s public/ folder:**
  ```bash
  cd /path/to/broadcaster_wrapper
  mkdir .\public # then copy the dist folder into here
  ```
3. **Run the producer server from .\src**
   ```bash
   python producer.py
   ```
4. **Connect from any machine in the network**
   ```bash
   https:\\<server-ip>:9003 # or whatever port the server is listening to
   ```
# Clock Synchronization

## ⏰ Synchronize Server-Client clocks via Public NTP Pools for Windows Machines

To get accurate one-way latency measurements, both machines must share the same time.

1. **Start & auto-enable the Time service**  
   ```bat
   sc config w32time start= auto
   net start w32time
   ```

2. **Point at the public NTP pool**
   ```bat
   w32tm /config /syncfromflags:manual /manualpeerlist:"0.pool.ntp.org,1.pool.ntp.org,2.pool.ntp.org" /reliable:yes /update
   ```
   
3. **Force an immediate resync & verify**
   ```bat
   w32tm /resync
   w32tm /query /status
   ```
Look for Source: 0.pool.ntp.org (or similar) and a valid Stratum.

## Linux ( Ubuntu )

### Server 

```
sudo apt update
sudo apt install ntp
```

```
sudo nano /etc/ntp.conf
sudo vim /etc/ntp.conf
```

Point to upstream servers of your choice (or leave the Ubuntu pools if you want this server to sync upstream too):

```
# comment these out and replace with close ones
pool 0.ubuntu.pool.ntp.org iburst
pool 1.ubuntu.pool.ntp.org iburst
```

Allow your client’s subnet (e.g. 192.168.1.0/24) to query this server—add somewhere in the file:

```
restrict 192.168.1.0 mask 255.255.255.0 nomodify notrap
```

Restart NTP

```
sudo systemctl restart ntp
sudo systemctl enable ntp
```


Verify the server is up: 

```
ntpq -p
```

### Client


```
sudo apt update
sudo apt install ntp
```

```
sudo nano /etc/ntp.conf
sudo vim /etc/ntp.conf
```

Comment out any existing pool or server lines, then add:

```
server 192.168.1.10 iburst
```

Check synchronization

```
ntpq -p
```

You should see the local server listed, and after a minute or two, a “reach” value >0 and the “offset” fairly small.

### Stop the service 


```
sudo systemctl disable ntp
sudo systemctl stop ntp
```


