import threading
import inspect
import broadcaster as bc

def setup_server():
    server = bc.ProducerServer(9002)

    #server.set_redirect("https://localhost:5173/")

    server.listen()

    return server