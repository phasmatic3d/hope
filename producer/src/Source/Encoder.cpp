#include "Encoder.h"

DracoEncoder::DracoEncoder(const DracoSettings& settings) {
    apply_settings(settings);
}

void DracoEncoder::apply_settings(const DracoSettings& settings) {
    settings.applyTo(m_encoder);
}

bool DracoEncoder::encode(const draco::PointCloud& point_cloud, EncodingStats& stats) {
    Timer t_encode;
    m_buffer.Clear();
    const draco::Status status = m_encoder.EncodePointCloudToBuffer(point_cloud, &m_buffer);
    stats.encode_ms = t_encode.elapsed_ms();
    if (!status.ok()) {
        std::cerr << "Encoding failed: " << status.error_msg_string() << std::endl;
        return false;
    }
    stats.encoded_bytes = m_buffer.size();
    return true;
}

draco::EncoderBuffer& DracoEncoder::GetBuffer() {
    return m_buffer;
}