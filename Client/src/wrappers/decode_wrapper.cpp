#include "draco/compression/decode.h"      
#include "draco/core/decoder_buffer.h"     
#include "draco/point_cloud/point_cloud.h" 

extern "C" {

    static constexpr int MAX_POINTS = 100000000;

    struct PointCloud {
        float*    positions;
        uint8_t*  colors;
        int       num_points;
    };

    
    static PointCloud g_out;

    static float    g_positions[MAX_POINTS * 3];
    static uint8_t  g_colors   [MAX_POINTS * 3];

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
        if (N > MAX_POINTS) {
        fprintf(stderr, "Too many points (%d > %d)\n", N, MAX_POINTS);
        return nullptr;
        }

        g_out.num_points = N;
        g_out.positions  = g_positions;
        g_out.colors     = g_colors;

        auto pos_att = pc->attribute(
            pc->GetNamedAttributeId(draco::GeometryAttribute::POSITION));
        auto col_att = pc->attribute(
            pc->GetNamedAttributeId(draco::GeometryAttribute::COLOR));

        draco::AttributeValueIndex avi;

        for (int i = 0; i < N; ++i) {
            avi = draco::AttributeValueIndex(i);
            pos_att->ConvertValue<float>(avi, g_positions  + 3*i);
            col_att->ConvertValue<uint8_t>(avi, g_colors + 3*i);
        }

        return &g_out;
    }

    // Free the struct and its buffers
    void free_pointcloud(PointCloud* pc) {
        if (!pc) return;
            free(pc->positions);
            free(pc->colors);
            free(pc);
        }

} // extern "C"