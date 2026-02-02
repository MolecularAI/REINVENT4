#FIXME: Running requires explicit port forwarding in vscode atleast for eduroam.

from tensorboard import program
import os

# Define the working directory
wd = f"{os.getcwd()}/runs"

# Set working directory to the latest subdirectory in runs
subdirs = [d for d in os.listdir(wd) if os.path.isdir(os.path.join(wd, d))]
latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(wd, d)))
wd = os.path.join(wd, latest_subdir)
print("Launching TensorBoard with logdir:", wd)

# Launch TensorBoard
tb = program.TensorBoard()
tb.configure(argv=[None, "--bind_all", "--logdir", wd, "--port", "8089"])
url = tb.launch()

print(f"TensorBoard is running at {url}")
input("Press Enter to terminate...")
