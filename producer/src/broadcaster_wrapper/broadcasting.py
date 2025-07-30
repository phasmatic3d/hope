from . import broadcaster 

def setup_server(
    port: int, 
    url: str,
    write_to_csv: bool=False
) -> broadcaster.ProducerServer:
    server = broadcaster.ProducerServer(port=port, write_to_csv=write_to_csv)
    server.set_redirect(url=url)
    return server


