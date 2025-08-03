import broadcaster as bc

def setup_server(
    port: int, 
    url: str,
    write_to_csv: bool,
    use_pings_for_rtt: bool,
) -> bc.ProducerServer:
    print(f"Write to CSV {write_to_csv}")
    print(f"Use Pins for RTT {use_pings_for_rtt}")
    print(f"URL: {url}")
    print(f"Port: {port}")
    server = bc.ProducerServer(port=port, write_to_csv=write_to_csv, use_pings_for_rtt=use_pings_for_rtt)
    #server.set_redirect(url=url)
    return server