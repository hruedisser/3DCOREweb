import sys
from coreweb.dashcore.app import app
import warnings
import os
import subprocess

def kill_process_on_port(port):
    # Define the shell script as a multi-line string
    shell_script = f"""#!/bin/bash

PORT={port}

# Get a list of PIDs using the specified port and kill them
for pid in $(lsof -ti :$PORT); do
  kill -9 $pid
done
"""

    # Create a temporary shell script file
    with open("kill_process.sh", "w") as script_file:
        script_file.write(shell_script)

    # Make the shell script executable
    subprocess.run(["chmod", "+x", "kill_process.sh"])

    # Execute the shell script
    subprocess.run(["./kill_process.sh"])

    # Remove the temporary shell script file
    os.remove("kill_process.sh")

def start_function():
    os.environ["OMP_NUM_THREADS"] = "1"  # Set the number of threads to 1 to use TBB
    warnings.filterwarnings("ignore")
    
    if len(sys.argv) == 2 and sys.argv[1] == "start":
        app.run_server(debug=True, port=8050)

if __name__ == "__main__":
    # Specify the port to be killed before starting the app
    port = 8050
    kill_process_on_port(port)
    start_function()