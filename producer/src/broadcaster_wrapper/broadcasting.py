from . import broadcaster 

def setup_server(
    port: int, 
    url: str,
) -> broadcaster.ProducerServer:
    print(f"URL: {url}")
    print(f"Port: {port}")
    server = broadcaster.ProducerServer(port=port)
    #server.set_redirect(url=url)
    return server


