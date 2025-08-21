#include "server.hpp"


ProducerServer::ProducerServer
(
    uint16_t port, 
    bool write_to_csv, 
    bool use_pings_for_rtt,
    spdlog::level::level_enum log_level
):
    m_port(port),
    write_to_csv(write_to_csv),
    use_pings_for_rtt(use_pings_for_rtt)
{
    m_logger = spdlog::stdout_color_mt("broadcaster");
    m_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");
    m_logger->set_level(log_level);
    m_logger->flush_on(log_level);
    m_logger->info("Initialized Broadcaster");

    m_logger->info("Initialized port: {}", m_port);
    m_logger->info("Write to CSV: {}", write_to_csv);
    m_logger->info("Use Pings for RTT: {}", use_pings_for_rtt);
    m_logger->info("Log Level: {}", spdlog::level::to_string_view(log_level));

    if(write_to_csv) start_logging_thread();

    m_server.init_asio();

    #ifdef USE_TLS
    m_server.set_tls_init_handler
    (
        [this](websocketpp::connection_hdl h) 
        {
            return this->tls_init_handler(h);
        }
    );
    #endif

    // disable logging
    m_server.clear_access_channels
    (
        websocketpp::log::alevel::frame_header   |
        websocketpp::log::alevel::frame_payload  |
        websocketpp::log::alevel::control
    );

    // On new WS connection
    m_server.set_open_handler
    (
        [this](websocketpp::connection_hdl hdl)
        {
            this->open_handler(hdl);
        }
    );

    // On WS close
    m_server.set_close_handler
    (
        [this](websocketpp::connection_hdl hdl)
        {
            this->close_handler(hdl);
        }
    );

    m_server.set_pong_handler
    (
        [this](websocketpp::connection_hdl h, const std::string& _)
        {
            this->pong_handler(h, _);
        }
    );

    m_server.set_message_handler
    (
        [this](websocketpp::connection_hdl hdl, websocketpp::server<websocketpp::config::asio>::message_ptr message)
        {
            try
            {
                this->message_handler(hdl, message);
            }
            catch(const std::exception& e)
            {
                m_logger->error("Error {} occurred while handling message", e.what());
            }
        }
    );

};

ProducerServer::~ProducerServer()
{
    if (write_to_csv)
    {
        // Signal logging thread to exit, then join it
        {
            std::lock_guard<std::mutex> lg(log_mutex);
            m_logger->debug("Acquired log mutex in: destructor");
            stop_logging = true;
            write_to_csv = false;
        }
        m_logger->debug("Unlocked log mutex in: desctructor");
        log_cv.notify_all();
        entry_cv.notify_all();
        if (log_thread.joinable()) log_thread.join();
    }
}

/**
 * -------------------------------------------
 *             PUBLIC METHODS
 * -------------------------------------------
**/

// Start listening on the given port
void ProducerServer::listen()
{
    m_server.listen(m_port);
    m_server.start_accept();
}

// Blocking run (typically spawn in a thread)
void ProducerServer::run()
{
    m_server.run();
}

void ProducerServer::stop()
{
    // Stop accepting any more incoming connections
    m_server.stop_listening();
    // Interrupt the ASIO loop so run() will return
    m_server.stop();
    if (write_to_csv)
    {
        // Signal logging thread to exit, then join it
        {
            std::lock_guard<std::mutex> lg(log_mutex);
            m_logger->debug("Acquired log mutex in: stop");
            stop_logging = true;
            write_to_csv = false;
        }
        m_logger->debug("Unlocked log mutex in: stop");
        log_cv.notify_all();
        entry_cv.notify_all();
        if (log_thread.joinable()) log_thread.join();
    }
    m_logger->info("Server stopped");
}

//void ProducerServer::broadcast(const nb::bytes& data)
void ProducerServer::broadcast(const void* data, std::size_t size)
{
    size_t broadcast_round  = 0;
    size_t message_size     = 0;
    size_t connections_size = 0;
    //TODO this assumes the bytes are received by all connections with no errors
    size_t bytes_sent       = 0;

    std::vector<websocketpp::connection_hdl> conns;
    {
        std::lock_guard<std::mutex> lock(m_connection_mutex);
        if (m_connections.empty()) return;
        broadcast_round  = m_broadcast_counter++;
        connections_size = m_connections.size();
        message_size     = size;
        bytes_sent       = message_size * connections_size;

        conns.assign(m_connections.begin(), m_connections.end());
    }

    std::map<MetaKey, PendingMetadata, MetaKeyCompare> map_to_metadata; 

    for (auto hdl: conns)
    {
        if(write_to_csv && !use_pings_for_rtt) 
        {
            auto error_code = _send_broadcast_info_packet(hdl, broadcast_round, message_size);
            if(error_code)
            {
                m_logger->warn("Broadcast info packet failed: {}", error_code.message()); 
                continue;
            }
        }

        auto error_code = this->_send_broadcast_data(hdl, data, size);

        if(error_code)
        {
            m_logger->warn("Broadcast raw packet failed: {}", error_code.message()); 
            continue;
        }


        if(write_to_csv && !use_pings_for_rtt)
        {
            std::ostringstream ss;
            ss << hdl.lock().get();

            m_logger->debug("Inserting metadata for broadcast round: {} and for connection: {}", broadcast_round, ss.str());
            m_metadata[{hdl, broadcast_round}] = PendingMetadata
            {
                hope::Clock::now(),
                broadcast_round,
                m_current_batch_id,
                message_size,  
                connections_size, 
                error_code ? error_code.message() : "no error"
            };
        }

    }

    if(!m_connections.empty() && write_to_csv && use_pings_for_rtt)
    {
        ping_async(broadcast_round, message_size, connections_size);
    }
}

void ProducerServer::set_redirect(const std::string &url)
{
    // Trim leading / trailing whitespace
    // std::string tmp = url.cast<std::string>();
    std::string tmp = url;
    auto ws_front = tmp.find_first_not_of(" \t\r\n");
    auto ws_back = tmp.find_last_not_of(" \t\r\n");
    if (ws_front == std::string::npos)
    {
        throw std::invalid_argument("Redirect URL cannot be empty or whitespace");
    }
    tmp = tmp.substr(ws_front, ws_back - ws_front + 1);

    // Strip a single trailing slash for parsing
    bool had_slash = (!tmp.empty() && tmp.back() == '/');
    if (had_slash)
        tmp.pop_back();

    // Lower-case prefix to check for scheme
    std::string lower = tmp;
    std::transform
    (
        lower.begin(),
        lower.end(),
        lower.begin(),
        [](unsigned char c)
        {
            return std::tolower(c);
        }
    );

    // accept http, https, ws, or wss
    if (lower.rfind("http://",0) == 0 ||
        lower.rfind("https://",0) == 0 ||
        lower.rfind("ws://",0)   == 0 ||
        lower.rfind("wss://",0)  == 0)
    {
        const auto scheme_end = tmp.find("://") + 3;
        const auto path_pos   = tmp.find('/', scheme_end);

        // grab “host[:port]”
        std::string authority = tmp.substr(
            scheme_end,
            (path_pos == std::string::npos ? tmp.size() : path_pos) - scheme_end
        );

        std::string rest = (path_pos == std::string::npos
                            ? std::string{}
                            : tmp.substr(path_pos));

        // if no “:port” add the server’s port
        if (authority.find(':') == std::string::npos) {
            authority += ":" + std::to_string(m_port);
        }

        // rebuild and re‐append the trailing slash
        m_redirect_url = tmp.substr(0, scheme_end)
                        + authority
                        + rest
                        + "/";
        m_logger->info("Constructed URL: {}", m_redirect_url);
        return;
    }

    // Otherwise treat as host[:port]
    std::string host;
    int port = m_port; // default to constructor port
    auto colon = tmp.find(':');
    if (colon != std::string::npos)
    {
        host = tmp.substr(0, colon);
        std::string port_str = tmp.substr(colon + 1);
        try
        {
            port = std::stoi(port_str);
        }
        catch (...)
        {
            m_logger->error("Invalid port in redirect URL: “" + port_str + "”");
            throw std::invalid_argument("Invalid port in redirect URL: “" + port_str + "”");
        }
    }
    else
    {
        host = tmp;
    }

    // Build final URL
    m_redirect_url = "http://" + host + ":" + std::to_string(port) + "/";
    m_logger->info("Constructed URL: {}", m_redirect_url);
}

CsvFileEntry ProducerServer::get_entry_for_round(const size_t broadcast_round)
{
    std::lock_guard<std::mutex> lg(log_mutex);
    return broadcast_round_to_entry[broadcast_round];
}

std::optional<CsvFileEntry> ProducerServer::wait_for_entry(const size_t broadcast_round) 
{
    if (!write_to_csv || m_connections.empty()) 
    {
        m_logger->debug("Early returning from: {}", __func__);
        return std::nullopt;
    } 

    m_logger->debug("Waiting for csv file entry in {}", __func__);
    std::unique_lock<std::mutex> lk(log_mutex);
    m_logger->debug("Acquired log mutex in: wait_for_entry");
    auto pop_if_present = [this](size_t round) -> std::optional<CsvFileEntry> 
    {
        auto it = broadcast_round_to_entry.find(round);
        if (it == broadcast_round_to_entry.end()) return std::nullopt;

        CsvFileEntry value = std::move(it->second);
        broadcast_round_to_entry.erase(it);
        m_logger->debug("Returning entry for round {}", round);
        return value;
    };


    if (auto hit = pop_if_present(broadcast_round))
    {
        return hit;
    }

    m_logger->debug("Will wait for entry for broadcast round: {}", broadcast_round);
    entry_cv.wait
    (
        lk, [this, broadcast_round] 
        {
            return broadcast_round_to_entry.count(broadcast_round) != 0 || stop_logging;
        }
    );

    m_logger->debug("Woke up in {}", __func__);

    return pop_if_present(broadcast_round);
}

void ProducerServer::set_current_batch_id(size_t id) 
{
    m_logger->debug("Setting batch ID: {}", id);
    m_current_batch_id = id;
}

std::size_t ProducerServer::connection_count() const 
{
    return m_connections.size();
}

//websocketpp::lib::error_code ProducerServer::_send_broadcast_data(websocketpp::connection_hdl hdl, const nb::bytes& data)
websocketpp::lib::error_code ProducerServer::_send_broadcast_data(websocketpp::connection_hdl hdl, const void* data, size_t size)
{
    websocketpp::lib::error_code ec;
    m_server.send
    (
        hdl,
        data,
        size,
        websocketpp::frame::opcode::binary,
        ec
    );

    if (ec) m_logger->error("Failed to send info JSON: {}", ec.message());
    return ec;

}

websocketpp::lib::error_code ProducerServer::_send_broadcast_info_packet(websocketpp::connection_hdl hdl, const size_t broadcast_round, const size_t message_size)
{
    websocketpp::lib::error_code ec;
    auto now = hope::Clock::now();
    auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    BroadcastInfo info;
    info.set_type("broadcast-info");
    info.set_message_size(message_size);
    info.set_broadcast_round(broadcast_round);
    info.set_send_timestamp_ms(epoch_ms);
    
    auto info_string = info.to_json_dump();

    m_logger->debug("Sending broadcast-info message to client: {}", info_string);

    m_server.send
    (
        hdl,
        info_string,
        websocketpp::frame::opcode::text,
        ec
    );

    if (ec) m_logger->error("Failed to send info JSON: {}", ec.message());
    return ec;
}

/**
 * -------------------------------------------
 *             WEBSOCKET HANDLERS
 * -------------------------------------------
**/

websocketpp::lib::shared_ptr<websocketpp::lib::asio::ssl::context> ProducerServer::tls_init_handler(websocketpp::connection_hdl hdl)
{
    namespace asio = websocketpp::lib::asio;
    try 
    {
        auto ctx = std::make_shared<asio::ssl::context>(asio::ssl::context::tls_server);

        ctx->set_options
        (
            asio::ssl::context::default_workarounds |
            asio::ssl::context::no_sslv2 |
            asio::ssl::context::no_sslv3 |
            asio::ssl::context::no_tlsv1 |
            asio::ssl::context::no_tlsv1_1 |
            asio::ssl::context::single_dh_use
        );

        // dev-only; for mTLS use verify_peer and load verify paths
        ctx->set_verify_mode(asio::ssl::verify_none);

        ctx->use_certificate_chain_file("./broadcaster_wrapper/cert/server.crt");
        ctx->use_private_key_file("./broadcaster_wrapper/cert/server.key", asio::ssl::context::pem);

        return ctx;
    } 
    catch (const std::exception& e) 
    {
        // log and abort the handshake
        spdlog::error("TLS init failed: {}", e.what());
        return {}; // causes the connection to fail TLS setup
    }
}

void ProducerServer::http_handler(websocketpp::connection_hdl hdl)
{
    auto con = m_server.get_con_from_hdl(hdl);

    auto base = std::filesystem::path("broadcaster_wrapper") / "public";
    m_logger->info("Serving files from {}", base.string());

    std::string req = con->get_resource();  // was get_request_uri()
    if (req == "/") req = "/index.html";                // default

    std::filesystem::path file = base / req.substr(1);
    if (!std::filesystem::exists(file) || std::filesystem::is_directory(file)) 
    {
        con->set_status(websocketpp::http::status_code::not_found);
        return;
    }

    // read the file
    std::ifstream in(file, std::ios::binary);
    std::ostringstream buf;
    buf << in.rdbuf();
    std::string body = buf.str();

    // set some basic headers
    con->set_status(websocketpp::http::status_code::ok);
    // very minimal MIME-type mapping
    if (file.extension() == ".html") 
    {
        con->replace_header("Content-Type", "text/html");
    } 
    else if (file.extension() == ".js") 
    {
        con->replace_header("Content-Type", "application/javascript");
    } 
    else if (file.extension() == ".css") 
    {
        con->replace_header("Content-Type", "text/css");
    }
    con->set_body(body);

}

void ProducerServer::open_handler(websocketpp::connection_hdl hdl)
{
    std::lock_guard<std::mutex> lock(m_connection_mutex);
    m_connections.insert(hdl);
    m_logger->info("Client connected");
}

void ProducerServer::close_handler(websocketpp::connection_hdl hdl)
{
    {
        std::lock_guard<std::mutex> lock(m_connection_mutex);
        m_connections.erase(hdl);
    }
    {
        std::lock_guard<std::mutex> lg(m_metadata_mutex);
        for (auto it = m_metadata.begin(); it != m_metadata.end();) 
        {
            if (it->first.first.lock().get() == hdl.lock().get()) it = m_metadata.erase(it);
            else ++it;
        }
    }
    {
        std::lock_guard<std::mutex> lg(ping_mutex);
        pending_pings.erase(hdl);
    }
    m_logger->info("Client disconnected");
}

void ProducerServer::pong_handler(websocketpp::connection_hdl hdl, const std::string& _)
{

    if(!this->write_to_csv || !this->use_pings_for_rtt)
    {
        m_logger->debug("Early return from {}.", __func__);
        return;
    }
    m_logger->debug("Received message in {}", __func__);

    PendingPing metadata;
    {
        std::lock_guard<std::mutex> lg(ping_mutex);
        auto it = pending_pings.find(hdl);
        if (it == pending_pings.end()) return;
        metadata = it->second;
        pending_pings.erase(it);
    }

    std::ostringstream connection_id;
    connection_id << hdl.lock().get();

    auto now = hope::Clock::now();
    double rtt = std::chrono::duration<double,std::milli>(now - metadata.send_time).count();

    struct { double rtt; } metrics{ rtt };

    CsvFileEntry entry;

    METRIC_CSV_RENAMES(APPLY_METRIC_RENAME, entry, metrics)
    CSV_ONLY_ASSIGNMENTS(APPLY_CSV_ONLY, entry, metrics)

    m_logger->debug("Received entry: {} in: {}", entry.create_csv_entry(), __func__);

    enqueue_log_entry(entry);
}

void ProducerServer::message_handler(websocketpp::connection_hdl hdl, websocketpp::server<websocketpp::config::asio>::message_ptr message)
{

    if(!this->write_to_csv || this->use_pings_for_rtt) 
    {
        m_logger->debug("Early return from {}.", __func__);
        return;
    }

    m_logger->debug("Received message: {}, in {}", message->get_payload(), __func__);

    auto parsed_json = nlohmann::json::parse(message->get_payload());
    if (parsed_json.value("type","") != "ms-and-processing")
    {
        m_logger->debug("The message cannot be processed...");
        return;
    }

    uint64_t client_ms = parsed_json.at("timestamp").get<uint64_t>();
    auto client_timestamp = hope::Timestamp
    (
        std::chrono::milliseconds(client_ms)
    );

    auto round = parsed_json.at("round").get<size_t>();
    std::ostringstream connection_id;
    connection_id << hdl.lock().get();
    MetaKey key{ hdl, round };
    PendingMetadata metadata; 
    {
        std::lock_guard<std::mutex> lg(m_metadata_mutex);
        auto it = m_metadata.find(key);
        if (it == m_metadata.end()) 
        {
            m_logger->warn("No metadata for hdl={} round={}", connection_id.str(), round);
            // Create a zeroed‐out entry so wait_for_entry() unblocks
            CsvFileEntry dummy;
            enqueue_log_entry(dummy);
            return;
        }
        metadata = it->second;
        m_metadata.erase(it);
    }

    ClientSuppliedMetrics client_supplied_metrics = ClientSuppliedMetrics::from_json_and_times(parsed_json, client_timestamp, metadata.send_time);
    m_logger->debug("Received Client supplied metrics:\n{}", client_supplied_metrics.to_string());

    CsvFileEntry entry;

    POPULATE_CSV_ENTRY(entry, client_supplied_metrics); 
    enqueue_log_entry(entry);
}

/**
 * -------------------------------------------
 *             PRIVATE METHODS
 * -------------------------------------------
**/

void ProducerServer::start_logging_thread()
{
    // open CSV file and write header
    const std::string path = "./broadcaster_wrapper/send_times.csv";

    bool need_header = true;
    {
        std::ifstream in(path);
        if (in.is_open())
        {
            std::string first;
            if (std::getline(in, first) && first == HEADER)
            {
                need_header = false;
            }
        }
        in.close();
    }

    csv.open(path, std::ios::out | std::ios::app);

    if (!csv.is_open())
    {
        m_logger->error("Failed to open log CSV at {}", path);
        throw std::runtime_error("Failed to open log CSV at " + path);
    }

    if (need_header)
    {
        m_logger->info("Writing header {} to csv", HEADER);
        csv << CsvFileEntry::header() << "\n";
    }

    // spawn background thread
    log_thread = std::thread([this]{
        std::vector<CsvFileEntry> batch;
        std::unique_lock<std::mutex> lk(log_mutex);
        m_logger->debug("Acquired log mutex in start_logging_thread...");
        for(;;)
        {

            log_cv.wait
            (
                lk, [this]
                {
                    return !log_queue.empty() || stop_logging;
                }
            );
            m_logger->debug("Woke up from log_cv in start_logging_thread...");

            while (!log_queue.empty()) 
            {
                batch.emplace_back(std::move(log_queue.front()));
                log_queue.pop();
            }

            bool stop = stop_logging;

            lk.unlock();
            m_logger->debug("Unlocked log mutex in start_logging_thread...");

            for (auto &e : batch) 
            {
                auto csv_entry = e.create_csv_entry();
                m_logger->debug("Created CSV entry: {}", csv_entry);
                csv << csv_entry << '\n';
            }

            if (!batch.empty()) csv.flush();
            batch.clear();

            if (stop) break;

            lk.lock();

        };
    });
}

void ProducerServer::enqueue_log_entry(const CsvFileEntry &entry)
{
    {
        std::lock_guard<std::mutex> lg(log_mutex);
        m_logger->debug("Acquired log mutex in: enqueue_log_entry");
        broadcast_round_to_entry[entry.broadcast_round] = entry;
        entry_cv.notify_all();
        log_queue.push(entry);
    }
    m_logger->debug("Unlocked log mutex in: enqueue_log_entry");
    log_cv.notify_one();
}

void ProducerServer::ping_async
(
    size_t broadcast_round,
    size_t message_size, 
    size_t connections_size
)
{

    std::map<websocketpp::connection_hdl, PendingPing, std::owner_less<websocketpp::connection_hdl>> to_insert;

    for(auto& hdl: m_connections)
    {
        websocketpp::lib::error_code ec;
        m_server.ping(hdl, "", ec);
        to_insert[hdl] = PendingPing
        {
            hope::Clock::now(),
            broadcast_round,
            m_current_batch_id,
            message_size,
            connections_size,
            ec ? ec.message() : "no ping error"
        };
    }

    {
        std::lock_guard<std::mutex> lg(ping_mutex);
        for (auto& kv : to_insert) 
        {
            pending_pings[kv.first] = kv.second;
        }
    }

}