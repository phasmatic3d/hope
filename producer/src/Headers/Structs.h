#pragma once
#include <chrono>
#include <cstddef>

struct Timer {
    using clock = std::chrono::steady_clock;
    clock::time_point start;

    Timer() : start(clock::now()) {}
    void reset() { start = clock::now(); }

    /// Returns elapsed milliseconds since construction or last reset
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - start).count();
    }
};


struct EncodingStats {
    double prep_ms = 0.0;      // time to feed Draco builder
    double encode_ms = 0.0;    // time to run encoder.Encode...
    size_t bytes = 0;          // final buffer size

    void print() const {
        // Build the new text
        char buf[128];
        int  len = std::snprintf(buf, sizeof(buf),
            "Prep: %7.2f ms | Encode: %7.2f ms | Size: %8zu bytes",
            prep_ms, encode_ms, bytes);
        if (len < 0) len = 0;
        if (len > (int)sizeof(buf) - 1) len = sizeof(buf) - 1;
        buf[len] = '\0';

        // Pad with spaces up to last_length so we fully clear the old text
        static size_t last_length = 0;
        size_t        target_len = std::max<size_t>(last_length, len);
        std::string   out(buf);
        if (out.size() < target_len) {
            out.append(target_len - out.size(), ' ');
        }
        last_length = out.size();

        // Carriage return + text + flush
        std::cout << '\r' << out << std::flush;
    }
};

struct DracoSettings {
    int posQuant = 10;  // position quantization bits
    int colorQuant = 8;   // color quantization bits
    int speedEncode = 5;   // encoder speed “compression vs speed”
    int speedDecode = 10;  // decoder speed 

    // Called when you press Space:
    void applyTo(draco::Encoder& enc) const {
        enc.SetSpeedOptions(speedEncode, speedDecode);
        enc.SetAttributeQuantization(draco::GeometryAttribute::POSITION, posQuant);
        enc.SetAttributeQuantization(draco::GeometryAttribute::COLOR, colorQuant);
    }

    // For display
    std::string toString() const {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "Qpos:%2d Qcol:%2d Spd:%2d/%2d",
            posQuant, colorQuant, speedEncode, speedDecode
        );
        return buf;
    }
};