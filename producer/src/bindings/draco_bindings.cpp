#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <draco/compression/encode.h>
#include <draco/compression/decode.h>
#include <draco/point_cloud/point_cloud.h>
#include <draco/point_cloud/point_cloud_builder.h>
#include <draco/core/encoder_buffer.h>

namespace nb = nanobind;

NB_MODULE(encoder, m) {
	m.doc() = "Draco PointCloud encoder (builder + encode)";

	m.def("encode_pointcloud",
		[](nb::ndarray<float, nb::numpy, nb::shape<-1, 3>>    positions,
			nb::ndarray<uint8_t, nb::numpy, nb::shape<-1, 3>>    colors,
			int                                            pos_quant,
			int                                            col_quant,
			int                                            speed_encode,
			int                                            speed_decode) -> nb::bytes {
				const int N = int(positions.shape(0));

				draco::PointCloudBuilder builder;
				builder.Start(N);

				const int pos_att_id = builder.AddAttribute(
					draco::GeometryAttribute::POSITION, 3, draco::DataType::DT_FLOAT32);

				const int col_att_id = builder.AddAttribute(
					draco::GeometryAttribute::COLOR, 3, draco::DataType::DT_UINT8);

				// Feed each point’s data
				const float* pos_ptr = positions.data();
				const uint8_t* col_ptr = colors.data();

				builder.SetAttributeValuesForAllPoints(pos_att_id, pos_ptr, 0);
				builder.SetAttributeValuesForAllPoints(col_att_id, col_ptr, 0);

				// Finalize the PointCloud (allocates storage, etc)
				std::unique_ptr<draco::PointCloud> pc = builder.Finalize(false);

				// Encode
				draco::EncoderBuffer buffer;
				draco::Encoder encoder;
				encoder.SetSpeedOptions(speed_encode, speed_decode);
				encoder.SetAttributeQuantization(draco::GeometryAttribute::POSITION, pos_quant);
				encoder.SetAttributeQuantization(draco::GeometryAttribute::COLOR, col_quant);

				const draco::Status status = encoder.EncodePointCloudToBuffer(*pc, &buffer);;

				//Return as Python bytes
				//Return bytes(ptr, length)
				const void* data = buffer.data();
				return nb::bytes(data, buffer.size());
		},
		nb::arg("positions"),
		nb::arg("colors"),
		nb::arg("pos_quant"),
		nb::arg("col_quant"),
		nb::arg("speed_encode"),
		nb::arg("speed_decode"),
		R"pbdoc(
            Encode a point cloud by giving:
              - positions: float32[N,3]
              - colors   : uint8[N,3]
              - pos_quant, col_quant (quantization bits)
              - speed_encode, speed_decode (0–10)
            Returns: encoded buffer bytes.
        )pbdoc"
	);
}