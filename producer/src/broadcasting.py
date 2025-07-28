import threading
import inspect
import broadcaster

def setup_server(port):
    server = broadcaster.ProducerServer(port)

    #server.set_redirect("https://localhost:5173/")

    server.listen()

    return server