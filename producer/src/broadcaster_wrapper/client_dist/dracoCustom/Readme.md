# DracoDecoder WASM Build Guide

This guide explains how to compile a custom C++ decoder into WebAssembly, and generate the accompanying JavaScript glue (`draco_decoder.js` / `draco_decoder.wasm`).  It assumes a Windows environment.

---

## Prerequisites

  **Emscripten SDK (emsdk)**
   - Download/clone from [https://github.com/emscripten-core/emsdk](https://github.com/emscripten-core/emsdk)
   - Install and activate
     ```bash
     git clone https://github.com/emscripten-core/emsdk.git
     cd emsdk
     ./emsdk install latest
     ./emsdk activate latest
     ./emsdk_env.bat      # (Windows) or `source ./emsdk_env.sh` on Unix
     ```


---

## 1. Fetch Draco Source

In your project’s `deps/` folder:

```bash
cd deps
git clone https://github.com/google/draco.git --branch v1.5.7 draco
```

---

## 2. Create a Emscripten Build Directory

```bash
cd draco
mkdir build_wasm && cd build_wasm
```

---

## 3. Configure & Build with Emscripten
```powershell
emcc `
-O3 `
-std=c++11 `
-I D:/Work/Work/Hope-new/hope/Client/deps/draco/src `
-I D:/Work/Work/Hope-new/hope/Client/deps/draco/build_wasm `
decode_wrapper.cpp `
D:/Work/Work/Hope-new/hope\Client/deps/draco/build_wasm/libdraco.a `
-s WASM=1 `
-s MODULARIZE=1 `
-s EXPORT_NAME="DracoDecoderModule" `
-s "EXPORTED_FUNCTIONS=['_decode_draco','_free_pointcloud', '_malloc', '_free' ]" `
-s "EXPORTED_RUNTIME_METHODS=['getValue','HEAPU8','HEAPF32']" `
-s ALLOW_MEMORY_GROWTH=1 `
-o draco_decoder.js

```
This should produce:

- `draco_decoder.js`
- `draco_decoder.wasm`

Copy these into the client’s `/public/dracoCustom/`  folder.

You can now run the decoder module on the client


