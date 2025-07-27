import threading
import inspect
import broadcaster

def setup_server():
    server = broadcaster.ProducerServer(9002)

    #server.set_redirect("https://localhost:5173/")

    server.listen()

    return server