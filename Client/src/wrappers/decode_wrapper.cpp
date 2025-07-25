#include "draco/compression/decode.h"      
#include "draco/core/decoder_buffer.h"     
#include "draco/point_cloud/point_cloud.h" 

extern "C" {

    struct PointCloud {
        float*    positions;
        uint8_t*  colors;
        int       num_points;
    };

    PointCloud* decode_draco(const uint8_t* data, size_t length) {
        draco::DecoderBuffer buffer;
        buffer.Init(reinterpret_cast<const char*>(data), length);

        draco::Decoder decoder;
        auto pc_or = decoder.DecodePointCloudFromBuffer(&buffer);
        if (!pc_or.ok()) {
            fprintf(stderr, "Draco decode error: %s\n",
                    pc_or.status().error_msg_string().c_str());
            return nullptr;
        }
        std::unique_ptr<draco::PointCloud> pc = std::move(pc_or).value();

        const int N = pc->num_points();
        int pos_att_id = pc->GetNamedAttributeId(draco::GeometryAttribute::POSITION);
        auto pos_att = pc->attribute(pos_att_id);

        int col_att_id = pc->GetNamedAttributeId(draco::GeometryAttribute::COLOR);
        auto col_att = pc->attribute(col_att_id);

        // Allocate output
        auto out = (PointCloud*)malloc(sizeof(PointCloud));
        out->num_points = N;
        out->positions  = (float*)malloc(sizeof(float)*3*N);
        out->colors     = (uint8_t*)malloc(sizeof(uint8_t)*3*N);

        draco::AttributeValueIndex avi(0);
        for (int i = 0; i < N; ++i) {
            avi = draco::AttributeValueIndex(i);
            pos_att->ConvertValue<float>(avi, out->positions  + 3*i);
            col_att->ConvertValue<uint8_t>(avi, out->colors + 3*i);
        }

        return out;
    }

    // Free the struct and its buffers
    void free_pointcloud(PointCloud* pc) {
        if (!pc) return;
            free(pc->positions);
            free(pc->colors);
            free(pc);
        }

} // extern "C"