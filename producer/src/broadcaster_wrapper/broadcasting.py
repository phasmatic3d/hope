from . import broadcaster 

def setup_server(
    port: int, 
    url: str
) -> broadcaster.ProducerServer:
    server = broadcaster.ProducerServer(port=port, write_to_csv=True)
    server.set_redirect(url=url)
    return server


