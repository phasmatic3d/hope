#pragma once

#include "Structs.h"
#include <draco/compression/encode.h>


namespace draco {
    class PointCloud;
    class EncoderBuffer;
}

class DracoEncoder {
public:
    explicit DracoEncoder(const DracoSettings& settings);
    bool encode(const draco::PointCloud& point_cloud,
        EncodingStats& stats);
    draco::EncoderBuffer& GetBuffer();

private:
    draco::Encoder m_encoder;
    draco::EncoderBuffer m_buffer;
    void apply_settings(const DracoSettings& settings);
};
