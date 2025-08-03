
from broadcasting import *
import hope_server
import torch
import os
import argparse

import producer_cli as producer_cli
import torch.multiprocessing as mp

from pathlib import Path
import threading
# Set up the server

def main():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")

    args = producer_cli.producer_cli.parse_args()

    server = setup_server(
        args.server_port,
        args.server_host,
        args.server_write_to_csv,
        args.server_use_pings_for_rtt
    )
    server.listen()
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    
    hope_server.launch_processes(server, args, DEVICE)
        

if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("Compiled CUDA version:", torch.version.cuda)
    print("Is CUDA available:", torch.cuda.is_available())
    main()