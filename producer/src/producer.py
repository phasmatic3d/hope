
from encoding import *


# Set up the server


def main():
    server = setup_server()
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    
    encode_point_cloud(server)

if __name__ == "__main__":
    main()