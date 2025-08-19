#define METRIC_CSV_COMMON_FIELDS(APPLY, entry, metrics) \
  APPLY(entry, metrics, one_way_ms)                     \
  APPLY(entry, metrics, one_way_plus_processing)        \
  APPLY(entry, metrics, wait_in_queue)                  \
  APPLY(entry, metrics, pure_decode_ms)                 \
  APPLY(entry, metrics, pure_geometry_upload_ms)        \
  APPLY(entry, metrics, pure_render_ms)                 \
  APPLY(entry, metrics, pure_processing_ms)             \
  APPLY(entry, metrics, chunk_decode_times)

// Same-name fields present in BOTH ClientSuppliedMetrics and CsvFileEntry
#define METRIC_CSV_RENAMES(APPLY, entry, metrics)       \
  APPLY(entry, metrics, rtt, approximate_rtt_ms)

// CSV-only fields and where to get their values from in the handler scope.
#define CSV_ONLY_ASSIGNMENTS(APPLY, entry, metrics)                  \
  APPLY(entry, metrics, timestamp,        metadata.send_time)        \
  APPLY(entry, metrics, connection_id,    connection_id.str())       \
  APPLY(entry, metrics, error,            std::move(metadata.error)) \
  APPLY(entry, metrics, broadcast_round,  metadata.round)            \
  APPLY(entry, metrics, batch_id,         metadata.batch_id)         \
  APPLY(entry, metrics, message_size,     metadata.message_size)     \
  APPLY(entry, metrics, connections_size, metadata.connections_size)


#define APPLY_METRIC_SAME(entry, metrics, NAME) \
  (entry).set_##NAME((metrics).NAME);

#define APPLY_METRIC_RENAME(entry, metrics, SRC, DST) \
  (entry).set_##DST((metrics).SRC);

#define APPLY_CSV_ONLY(entry, metrics, FIELD, EXPR) \
  (entry).set_##FIELD(EXPR);

#define POPULATE_CSV_ENTRY(entry, metrics)                      \
  do {                                                          \
    METRIC_CSV_COMMON_FIELDS(APPLY_METRIC_SAME, entry, metrics) \
    METRIC_CSV_RENAMES(APPLY_METRIC_RENAME, entry, metrics)     \
    CSV_ONLY_ASSIGNMENTS(APPLY_CSV_ONLY, entry, metrics)        \
  } while (0);