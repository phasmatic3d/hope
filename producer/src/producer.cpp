#include <stdio.h>

#include <iostream>
#include <fstream>

#include "draco/compression/encode.h"
#include "draco/point_cloud/point_cloud.h"
#include "draco/point_cloud/point_cloud_builder.h"
#include "draco/core/encoder_buffer.h"

int main() {
    // Define some simple points (x, y, z)
    std::vector<float> point_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };

    const int num_points = 3;
    const int num_components = 3; // x, y, z

    // Create a point cloud builder
    draco::PointCloudBuilder builder;
    builder.Start(num_points);

    // Add position attribute
    int pos_att_id = builder.AddAttribute(draco::GeometryAttribute::POSITION, num_components, draco::DT_FLOAT32);
    builder.SetAttributeValuesForAllPoints(pos_att_id, point_data.data(), 0);


    // Build the point cloud
    std::unique_ptr<draco::PointCloud> pc = builder.Finalize(false);
    if (!pc) {
        std::cerr << "Failed to build point cloud.\n";
        return -1;
    }

    // Set encoding options
    draco::Encoder encoder;
    encoder.SetSpeedOptions(5, 5); // balance between speed and compression
    encoder.SetAttributeQuantization(draco::GeometryAttribute::POSITION, 14);

    // Encode the point cloud
    draco::EncoderBuffer buffer;
    const draco::Status status = encoder.EncodePointCloudToBuffer(*pc, &buffer);

    if (!status.ok()) {
        std::cerr << "Encoding failed: " << status.error_msg_string() << std::endl;
        return -1;
    }

    std::cout << "Successfully encoded point cloud! Size = " << buffer.size() << " bytes\n";

    // (Optional) Write to file
    std::ofstream outfile("point_cloud.drc", std::ios::binary);
    outfile.write(buffer.data(), buffer.size());
    outfile.close();

    printf("Hello world\n");

    return 0;
}