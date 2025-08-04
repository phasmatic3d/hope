# 🛠️ Local “localhost” TLS Setup for Your WebSocket Server

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

# Create a server key & CSR for “localhost”

## 3a. Server private key
```
openssl genrsa -out server.key 2048
```

## 3b. Certificate Signing Request (CSR)
```
openssl req -new \
  -key server.key \
  -out server.csr \
  -subj "/C=US/ST=State/L=City/O=Hope Dev/CN=localhost"
```

# Create the SAN extension file
Create a file named `server.ext` next to your CSR with these contents:


```
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1  = 127.0.0.1
IP.2  = ::1
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

# Install your CA into Ubuntu’s trust store ( I don't know the windows equiv. )
```
# Copy the CA cert into the system store
sudo cp myCA.pem /usr/local/share/ca-certificates/myCA.crt

# Update the CA bundle
sudo update-ca-certificates
```


# Test if the key is trusted by the system

```
export NODE_EXTRA_CA_CERTS="/usr/local/share/ca-certificates/myCA.crt"
```

```
wscat -c wss://localhost:9003       # should now connect
```

# Make the browser trust the key 

```
sudo apt install libnss3-tools
```


```
certutil -d sql:"$HOME/.pki/nssdb" \
        -A -t "C,," \
        -n "Local Dev Root CA" \
        -i ~/Github/hope/producer/src/broadcaster_wrapper/cert/myCA.pem
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


