#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <draco/compression/encode.h>
#include <draco/compression/decode.h>
#include <draco/point_cloud/point_cloud.h>
#include <draco/point_cloud/point_cloud_builder.h>
#include <draco/core/encoder_buffer.h>
#include <draco/core/decoder_buffer.h>

#include <cstring>  // for std::memcpy
#include <array>    // for std::array
#include <cstddef>

using ssize_t = std::ptrdiff_t;
namespace nb = nanobind;

NB_MODULE(draco_bindings, m) {
	m.doc() = "Draco PointCloud encoder (builder + encode)";

	m.def("encode_pointcloud",
		[](nb::ndarray<float, nb::numpy, nb::shape<-1, 3>>    positions,
			nb::ndarray<uint8_t, nb::numpy, nb::shape<-1, 3>>    colors,
			int                                            pos_quant,
			int                                            col_quant,
			int                                            speed_encode,
			int                                            speed_decode,
            bool                                           deduplicate) -> nb::bytes {
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
				std::unique_ptr<draco::PointCloud> pc = builder.Finalize(deduplicate);

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
        nb::arg("deduplicate"),
		R"pbdoc(
            Encode a point cloud by giving:
              - positions: float32[N,3]
              - colors   : uint8[N,3]
              - pos_quant, col_quant (quantization bits)
              - speed_encode, speed_decode (0–10)
            Returns: encoded buffer bytes.
        )pbdoc"
	);

	m.def("decode_pointcloud",
        [](nb::bytes encoded) {
            // 1) Feed Python bytes into Draco
            const char *draco_data = reinterpret_cast<const char*>(encoded.data());
            size_t      draco_size = encoded.size();
            draco::DecoderBuffer buffer;
            buffer.Init(draco_data, draco_size);

            // 2) Decode
            draco::Decoder decoder;
            auto status_or_pc = decoder.DecodePointCloudFromBuffer(&buffer);
            if (!status_or_pc.ok())
                throw std::runtime_error(status_or_pc.status().error_msg());
            std::unique_ptr<draco::PointCloud> pc = std::move(status_or_pc).value();
            int N = pc->num_points();

            // 3) POSITION: grab raw buffer pointer + byte size
            auto *pos_attr = pc->GetNamedAttribute(draco::GeometryAttribute::POSITION);
            if (!pos_attr) throw std::runtime_error("No POSITION attribute");
            const void *pos_buf_data = pos_attr->buffer()->data();
            size_t      pos_bytes    = pos_attr->buffer()->data_size();

            // 4) Allocate a  float[N*3], copy into it
            float *pos_copy = new float[N * 3];
            std::memcpy(pos_copy, pos_buf_data, pos_bytes);

            // 5) Wrap in a capsule so Python will delete[] it
            nb::capsule pos_owner(pos_copy, [](void *p) noexcept {
                delete[] static_cast<float*>(p);
            });

            // 6) Build the ndarray: a C-contiguous 2D float32 array, shape=(N,3)
            using PosArray = nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig>;
            PosArray positions(pos_copy, { (size_t)N, 3 }, pos_owner);

            // 7) Repeat for COLOR
            auto *col_attr = pc->GetNamedAttribute(draco::GeometryAttribute::COLOR);
            if (!col_attr) throw std::runtime_error("No COLOR attribute");
            const void   *col_buf_data = col_attr->buffer()->data();
            size_t        col_bytes    = col_attr->buffer()->data_size();

            uint8_t *col_copy = new uint8_t[N * 3];
            std::memcpy(col_copy, col_buf_data, col_bytes);

            nb::capsule col_owner(col_copy, [](void *p) noexcept {
                delete[] static_cast<uint8_t*>(p);
            });

            using ColArray = nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>, nb::c_contig>;
            ColArray colors(col_copy, { (size_t)N, 3 }, col_owner);

            // 8) Return the two NumPy arrays
            return nb::make_tuple(positions, colors);
        },
        nb::arg("encoded"),
        R"pbdoc(
            Decode a Draco-encoded point cloud.

            Parameters
            ----------
            encoded : bytes
                The byte string produced by `encode_pointcloud`.

            Returns
            -------
            positions : numpy.ndarray, shape (N,3), dtype float32
            colors    : numpy.ndarray, shape (N,3), dtype uint8
        )pbdoc"
    );

}