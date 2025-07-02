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
	double det_ms = 0.0; // time to run detection
	double pc_ms = 0.0; // time to find ROI PC
    double prep_ms = 0.0;   // time to feed Draco builder
    double encode_ms = 0.0;   // time to run encoder.Encode...
    double decode_ms = 0.0;   // time to run decoder.Decode...
    mutable double total_time_ms = 0.0;
    size_t encoded_bytes = 0;     // final buffer size
    size_t raw_bytes = 0; // encoded buffer size
	size_t pts = 0; // number of points in the point cloud

    void print() const {
        // On the very first call we need to emit blank lines to reserve space.
        static bool first = true;
        const int LINES = 8;  // total lines we will print each frame
        if (first) {
            for (int i = 0; i < LINES; ++i) std::cout << "\n";
            first = false;
        }

        // Move cursor back up so our new block overwrites the old one
        std::cout << "\033[" << LINES << "A";

        // compute savings %
        double savings = 0.0;
        if (raw_bytes > 0) {
            savings = 100.0 * (static_cast<double>(raw_bytes) - static_cast<double>(encoded_bytes))
                / static_cast<double>(raw_bytes);
        }

        // Now print each stat on its own line
        std::cout << "Obj Detection time : " << std::fixed << std::setprecision(2) << det_ms << " ms\n"
            << "PC   time : " << std::fixed << std::setprecision(2) << pc_ms << " ms\n"
            << "Prep  time : " << std::fixed << std::setprecision(2) << prep_ms << " ms\n"
            << "Encode time : " << std::fixed << std::setprecision(2) << encode_ms << " ms\n"
            << "Decode time : " << std::fixed << std::setprecision(2) << decode_ms << " ms\n"
            << "Pts         : " << pts << "\n"
            << "Raw/Enc     : "
            << raw_bytes << " B / " << encoded_bytes << " B"
            << "  (Saved: " << std::fixed << std::setprecision(1) << savings << "%)\n"
            << std::flush;
    }

    void printBodyOnly() const {
        // compute total_time
        total_time_ms = det_ms + pc_ms + prep_ms + encode_ms + decode_ms;

        double savings = (raw_bytes > 0)
            ? 100.0 * (raw_bytes - encoded_bytes) / raw_bytes
            : 0.0;
        std::cout
            << "Obj Detection time : " << std::fixed << std::setprecision(2) << det_ms << " ms\n"
            << "PC   time        : " << std::fixed << std::setprecision(2) << pc_ms << " ms\n"
            << "Prep  time        : " << std::fixed << std::setprecision(2) << prep_ms << " ms\n"
            << "Encode time       : " << std::fixed << std::setprecision(2) << encode_ms << " ms\n"
            << "Decode time       : " << std::fixed << std::setprecision(2) << decode_ms << " ms\n"
            << "Pts               : " << pts << "\n"
            << "Raw/Enc           : "
            << raw_bytes << " B / " << encoded_bytes << " B"
            << "  (Saved: " << std::fixed << std::setprecision(1) << savings << "%)\n";
    }
};

struct DracoSettings {
    int posQuant = 10;  // position quantization bits
    int colorQuant = 8;   // color quantization bits
    int speedEncode = 10;   // encoder speed “compression vs speed”
    int speedDecode = 10;  // decoder speed 
    int roiWidth = 240; // default ROI width
    int roiHeight = 240; // default ROI height



    void applyTo(draco::Encoder& enc) const {
        enc.SetSpeedOptions(speedEncode, speedDecode);
        enc.SetAttributeQuantization(draco::GeometryAttribute::POSITION, posQuant);
        enc.SetAttributeQuantization(draco::GeometryAttribute::COLOR, colorQuant);
    }


    // For display
    std::string toString() const {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "Qpos:%2d Qcol:%2d Spd:%2d/%2d ROI:%dx%d",
            posQuant, colorQuant, speedEncode, 10,
            roiWidth, roiHeight
        );
        return buf;
    }
};