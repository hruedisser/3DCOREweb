import sys
from coreweb.dashcore.app import app
import warnings
import os

def start_function():
    
    os.environ["OMP_NUM_THREADS"] = "1"  # Set the number of threads to 1 to use TBB

    warnings.filterwarnings("ignore")
    if len(sys.argv) == 2 and sys.argv[1] == "start":
        app.run_server(port=8899, debug=True)

if __name__ == "__main__":
    start_function()